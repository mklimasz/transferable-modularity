import json
from os import mkdir
from os.path import exists, isdir
from os.path import join
from typing import Sequence

import torch
from transformers.adapters.loading import AdapterLoader
from transformers.adapters.utils import logger, ACTIVATION_RENAME, WEIGHTS_NAME, CONFIG_NAME


class WeightsLoaderHelper:
    """
    A class providing helper methods for saving and loading module weights.
    """

    def __init__(self, model,
                 weights_name,
                 config_name,
                 load_every_k_layer: int = 4,
                 avg_layers: bool = False,
                 teacher_type: str = "roberta",
                 student_type: str = "distilbert"):
        self.model = model
        self.weights_name = weights_name
        self.config_name = config_name
        self.load_every_k_layer = load_every_k_layer
        self.avg_layers = avg_layers
        self.teacher_type = teacher_type
        self.student_type = student_type

    def state_dict(self, filter_func):
        return {k: v for (k, v) in self.model.state_dict().items() if filter_func(k)}

    def rename_state_dict(self, state_dict, *rename_funcs):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for rename_func in rename_funcs:
                new_k = rename_func(new_k)
            new_state_dict[new_k] = v
        return new_state_dict

    def save_weights_config(self, save_directory, config, meta_dict=None):
        # add meta information if given
        if meta_dict:
            for k, v in meta_dict.items():
                if k not in config:
                    config[k] = v
        # save to file system
        output_config_file = join(save_directory, self.config_name)
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        logger.info("Configuration saved in {}".format(output_config_file))

    def save_weights(self, save_directory, filter_func):
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), "Saving path should be a directory where the module weights can be saved."

        # Get the state of all adapter modules for this task
        state_dict = self.state_dict(filter_func)
        # Save the adapter weights
        output_file = join(save_directory, self.weights_name)
        torch.save(state_dict, output_file)
        logger.info("Module weights saved in {}".format(output_file))

    def load_weights_config(self, save_directory):
        config_file = join(save_directory, self.config_name)
        logger.info("Loading module configuration from {}".format(config_file))
        # Load the config
        with open(config_file, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        # For older versions translate the activation function to the new format
        if "version" not in loaded_config:
            if "config" in loaded_config and loaded_config["config"] is not None:
                if (
                        "non_linearity" in loaded_config["config"]
                        and loaded_config["config"]["non_linearity"] in ACTIVATION_RENAME
                ):
                    loaded_config["config"]["non_linearity"] = ACTIVATION_RENAME[
                        loaded_config["config"]["non_linearity"]
                    ]
        return loaded_config

    @staticmethod
    def _load_module_state_dict(module, state_dict, start_prefix=""):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(module, prefix=start_prefix)

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    module.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return missing_keys, unexpected_keys

    def load_weights(
            self,
            save_directory,
            filter_func,
            rename_func=None,
            loading_info=None,
            in_base_model=False,
    ):
        weights_file = join(save_directory, self.weights_name)
        # Load the weights of the adapter
        try:
            state_dict = torch.load(weights_file, map_location="cpu")
        except Exception:
            raise OSError("Unable to load weights from pytorch checkpoint file. ")

        # Rename weights if needed
        if rename_func:
            if isinstance(rename_func, Sequence):
                state_dict = self.rename_state_dict(state_dict, *rename_func)
            else:
                state_dict = self.rename_state_dict(state_dict, rename_func)

        custom_state_dict = {}

        logger.warning(f"Loading every {self.load_every_k_layer} layer!!!!!!!!!!!!! Averaging state: {self.avg_layers}")
        for key, value in state_dict.items():
            key = key.replace("_new.", "")
            layer_idx = int(key.split(".")[3])
            new_layer_idx = layer_idx // self.load_every_k_layer
            key_parts = key.split(".")
            merged_key = ".".join(key_parts[:3]) + "." + str(new_layer_idx) + "." + ".".join(key_parts[4:])
            merged_key = self.key_mapping(merged_key)
            if layer_idx % self.load_every_k_layer == 0:
                custom_state_dict[merged_key] = value
            elif layer_idx % self.load_every_k_layer < self.load_every_k_layer - 1 and self.avg_layers:
                custom_state_dict[merged_key] += value
            else:
                if self.avg_layers:
                    assert layer_idx % self.load_every_k_layer < self.load_every_k_layer
                    custom_state_dict[merged_key] = custom_state_dict[merged_key] / self.load_every_k_layer

        logger.info("Adding adapters:")
        for k in custom_state_dict.keys():
            logger.info(f"Key: {k} with shape: {custom_state_dict[k].size()}")

        state_dict = custom_state_dict

        logger.info("Loading module weights from {}".format(weights_file))

        # Add the weights to the model
        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = self.model
        has_prefix_module = any(s.startswith(self.model.base_model_prefix) for s in state_dict.keys())
        if not hasattr(self.model, self.model.base_model_prefix) and has_prefix_module:
            start_prefix = self.model.base_model_prefix + "."
        if in_base_model and hasattr(self.model, self.model.base_model_prefix) and not has_prefix_module:
            model_to_load = self.model.base_model

        missing_keys, unexpected_keys = self._load_module_state_dict(
            model_to_load, state_dict, start_prefix=start_prefix
        )

        missing_keys = [k for k in missing_keys if filter_func(k)]
        if len(missing_keys) > 0:
            logger.info(
                "Some module weights could not be found in loaded weights file: {}".format(", ".join(missing_keys))
            )
        if self.model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if k not in self.model._keys_to_ignore_on_load_unexpected]
        if len(unexpected_keys) > 0:
            logger.info(
                "Some weights of the state_dict could not be loaded into model: {}".format(", ".join(unexpected_keys))
            )

        if isinstance(loading_info, dict):
            if "missing_keys" not in loading_info:
                loading_info["missing_keys"] = []
            if "unexpected_keys" not in loading_info:
                loading_info["unexpected_keys"] = []
            loading_info["missing_keys"].extend(missing_keys)
            loading_info["unexpected_keys"].extend(unexpected_keys)

        return missing_keys, unexpected_keys

    def key_mapping(self, key):
        if self.student_type == "distilbert":
            assert self.teacher_type in {"roberta", "bert"}
            key = (key.replace("roberta.encoder", "transformer")
                   .replace("bert.encoder", "transformer")
                   .replace("output", "output_adapters")
                   .replace("self.query", "q_lin")
                   .replace("self.value", "v_lin"))
        elif self.student_type == "bert":
            assert self.teacher_type == "roberta"
            key = key.replace("roberta.encoder", "bert.encoder")
        elif self.student_type == "roberta":
            # Nothing to change
            pass
        else:
            raise NotImplementedError(f"Not implemented key mapping for student {self.student_type}")
        return key


class CustomAdapterLoader(AdapterLoader):

    def __init__(self, model, adapter_type=None,
                 load_every_k_layer: int = 4,
                 avg_layers: bool = False,
                 teacher_type: str = "roberta",
                 student_type: str = "distilbert"):
        super().__init__(model, adapter_type)
        self.weights_helper = WeightsLoaderHelper(model, WEIGHTS_NAME, CONFIG_NAME,
                                                  load_every_k_layer,
                                                  avg_layers,
                                                  teacher_type,
                                                  student_type)
