import functools
import json
import math
import os
from dataclasses import dataclass, field
from types import MethodType

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from sklearn.feature_selection import r_regression
from torch import nn
from transformers import AutoAdapterModel, AutoTokenizer, HfArgumentParser

from mkd.matrix import linear_max_sum_assignment


@dataclass
class PruningArguments:
    teacher_model_type: str = field(default="roberta", metadata={"help": ""})
    teacher_model_name: str = field(default="xlm-roberta-large", metadata={"help": ""})
    student_model_type: str = field(default="distilbert", metadata={"help": ""})
    student_model_name: str = field(default="distilbert-base-multilingual-cased", metadata={"help": ""})
    adapter_path: str = field(default="", metadata={"help": ""})
    adapter_type: str = field(default="", metadata={"help": ""})
    pruned_adapter_path: str = field(default="", metadata={"help": ""})
    dataset_name: str = field(default="", metadata={"help": ""})
    dataset_config_name: str = field(default=None, metadata={"help": ""})
    topk: int = field(default=1000, metadata={"help": ""})
    reduction_size: int = field(default=768, metadata={"help": ""})
    student_reduction_factor: int = field(default=6, metadata={"help": ""})
    prune_every_k: int = field(default=4, metadata={"help": ""})
    workers: int = field(default=32, metadata={"help": "Correlation calculation workers."})


if __name__ == '__main__':
    parser = HfArgumentParser(PruningArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    padding = True
    max_seq_length = None
    device = "cuda:0"

    teacher_model = AutoAdapterModel.from_pretrained(args.teacher_model_name)
    teacher_model.eval()
    teacher_model.to(device)

    student_model = AutoAdapterModel.from_pretrained(args.student_model_name)
    student_model.eval()
    student_model.to(device)

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_name,
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_model_name,
    )

    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
    )

    if args.dataset_name == "Babelscape/wikineural":
        raw_dataset = raw_datasets[f"train_{args.dataset_config_name}"]
    else:
        raw_dataset = raw_datasets["train"]
    prune_dataset = raw_dataset.select(range(args.topk))


    def tokenize(examples, tokenizer):
        if args.dataset_name == "paws-x":
            key1, key2 = "sentence1", "sentence2"
        elif args.dataset_name == "Babelscape/wikineural":
            return tokenizer(
                [x["tokens"] for x in examples],
                padding=padding,
                truncation=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
        else:
            key1, key2 = "premise", "hypothesis"
        return tokenizer(
            [x[key1] for x in examples],
            [x[key2] for x in examples],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )


    class ProbingMixin(nn.Module):
        def __init__(self, model, model_type, adapter_type):
            super().__init__()
            self.model = model
            self.model_type = model_type
            self.adapter_type = adapter_type

            self.install_hook(self.adapter_type)
            self.before = {}
            self.after = {}

            self.tmp_before = {}
            self.tmp_after = {}

        def install_hook(self, adapter_type):
            if adapter_type == "pfeiffer":
                self.install_pfeiffer_hook()
            elif adapter_type == "lora":
                self.install_lora_hook()
            else:
                raise NotImplementedError(f"Not implemented for adapter of type {self.adapter_type}")

        def install_lora_hook(self):
            layers = self._get_layers()
            for idx, layer in enumerate(layers):
                old_forward = self._get_lora_forward_method(layer)

                def probing_hook(idx, cls, hidden_states):
                    if idx not in self.tmp_before:
                        self.tmp_before[idx] = []
                    self.tmp_before[idx] = hidden_states.view(-1, hidden_states.size(-1)).cpu()

                    output = old_forward(hidden_states)

                    if idx not in self.tmp_after:
                        self.tmp_after[idx] = []
                    self.tmp_after[idx] = output.view(-1, output.size(-1)).cpu()

                    return output

                func = functools.partial(probing_hook, idx)
                self._set_lora_forward_method(func, layer)

        def install_pfeiffer_hook(self):
            layers = self._get_layers()

            for idx, layer in enumerate(layers):
                old_forward = self._get_adapter_forward_method(layer)

                def probing_hook(idx, cls, hidden_states, residual_input, layer_norm):
                    if idx not in self.tmp_before:
                        self.tmp_before[idx] = []
                    self.tmp_before[idx] = hidden_states.view(-1, hidden_states.size(-1)).cpu()

                    output = old_forward(hidden_states, residual_input, layer_norm)

                    if idx not in self.tmp_after:
                        self.tmp_after[idx] = []
                    self.tmp_after[idx] = output.view(-1, output.size(-1)).cpu()

                    return output

                func = functools.partial(probing_hook, idx)
                self._set_adapter_forward_method(func, layer)

        def _set_lora_forward_method(self, func, layer):
            if self.model_type in {"bert", "roberta"}:
                layer.attention.self.query.forward = MethodType(func, layer.attention.self.query)
            elif self.model_type == "distilbert":
                layer.attention.q_lin.forward = MethodType(func, layer.attention.q_lin)
            else:
                raise NotImplementedError(f"Not implemented for model of type {self.model_type}")

        def _set_adapter_forward_method(self, func, layer):
            if self.model_type in {"bert", "roberta"}:
                layer.output.adapter_layer_forward = MethodType(func, layer.output)
            elif self.model_type == "distilbert":
                layer.output_adapters.adapter_layer_forward = MethodType(func, layer.output_adapters)
            else:
                raise NotImplementedError(f"Not implemented for model of type {self.model_type}")

        def _get_lora_forward_method(self, layer):
            if self.model_type in {"bert", "roberta"}:
                forward = layer.attention.self.query.forward
            elif self.model_type == "distilbert":
                forward = layer.attention.q_lin.forward
            else:
                raise NotImplementedError(f"Not implemented for model of type {self.model_type}")
            return forward

        def _get_adapter_forward_method(self, layer):
            if self.model_type in {"bert", "roberta"}:
                forward = layer.output.adapter_layer_forward
            elif self.model_type == "distilbert":
                forward = layer.output_adapters.adapter_layer_forward
            else:
                raise NotImplementedError(f"Not implemented for model of type {self.model_type}")
            return forward

        def _get_layers(self):
            if self.model_type in {"bert", "roberta"}:
                layers = getattr(self.model, self.model_type).encoder.layer
            elif self.model_type == "distilbert":
                layers = self.model.distilbert.transformer.layer
            else:
                raise NotImplementedError(f"Not implemented for model of type {self.model_type}")
            return layers

        def forward(self, *args, **kwargs):
            _ = self.model(*args, **kwargs)

        def add_with_filtering(self, mask):
            for layer_idx, embeddings in self.tmp_before.items():
                if layer_idx not in self.before:
                    self.before[layer_idx] = []
                assert embeddings.size(0) == len(mask), f"{embeddings.size()}, {len(mask)}"
                for emb, m in zip(embeddings, mask):
                    if m == 1:
                        self.before[layer_idx].append(emb)

            for layer_idx, embeddings in self.tmp_after.items():
                if layer_idx not in self.after:
                    self.after[layer_idx] = []
                assert embeddings.size(0) == len(mask), f"{embeddings.size()}, {len(mask)}"
                for emb, m in zip(embeddings, mask):
                    if m == 1:
                        self.after[layer_idx].append(emb)

            self.tmp_before = {}
            self.tmp_after = {}


    teacher_probing_model = ProbingMixin(teacher_model, args.teacher_model_type, args.adapter_type).eval()
    student_probing_model = ProbingMixin(student_model, args.student_model_type, args.adapter_type).eval()


    def convert_roberta_to_prefix_tokens(token_ids, tokenizer, model_type):
        subwords = tokenizer.convert_ids_to_tokens(token_ids)
        if model_type == "roberta":
            preprocessed = []
            for subword in subwords:
                if subword[0] == "▁":
                    preprocessed.append(subword[1:])
                else:
                    preprocessed.append("##" + subword)
            return preprocessed
        else:
            return subwords


    def batchify(seq, batch_size: int = 16):
        batch = []
        for el in seq:
            batch.append(el)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


    with torch.no_grad():
        batch_size = 8
        for sample in tqdm.tqdm(batchify(prune_dataset, batch_size),
                                total=int(math.ceil(len(prune_dataset) / batch_size)),
                                desc="Tokens sampling."):
            tokenized_sample = tokenize(sample, teacher_tokenizer)

            teacher_tokens = [convert_roberta_to_prefix_tokens(tokenized_sample["input_ids"][x],
                                                               teacher_tokenizer,
                                                               args.teacher_model_type) for x in range(len(sample))]
            tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items()}
            teacher_probing_model.forward(**tokenized_sample)

            tokenized_sample = tokenize(sample, student_tokenizer)
            student_tokens = [convert_roberta_to_prefix_tokens(tokenized_sample["input_ids"][x],
                                                               student_tokenizer,
                                                               args.student_model_type) for x in range(len(sample))]
            tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items()}
            student_probing_model.forward(**tokenized_sample)

            batch_teacher_token_mask = []
            batch_student_token_mask = []
            for x in range(len(sample)):

                teacher_token_mask = [0 for _ in teacher_tokens[x]]
                student_token_mask = [0 for _ in student_tokens[x]]
                for idx, (s, t) in enumerate(zip(student_tokens[x], teacher_tokens[x])):
                    if s == t and tokenized_sample["input_ids"][x][idx] != student_tokenizer.pad_token_id:
                        teacher_token_mask[idx] = 1
                        student_token_mask[idx] = 1
                print("Shared tokens for batch", sum(teacher_token_mask))

                assert len(teacher_token_mask) == len(teacher_tokens[x])
                assert len(student_token_mask) == len(student_tokens[x])
                batch_teacher_token_mask.extend(teacher_token_mask)
                batch_student_token_mask.extend(student_token_mask)

            teacher_probing_model.add_with_filtering(batch_teacher_token_mask)
            student_probing_model.add_with_filtering(batch_student_token_mask)


    def get_down_project_key(model_type, layer_idx, adapter_type):
        if adapter_type == "pfeiffer":
            return f"{model_type}.encoder.layer.{layer_idx}.output.adapters.ner.adapter_down.0.weight", None
        elif adapter_type == "lora":
            return (f"{model_type}.encoder.layer.{layer_idx}.attention.self.query.loras.ner.lora_A",
                    f"{model_type}.encoder.layer.{layer_idx}.attention.self.value.loras.ner.lora_A")
        else:
            raise NotImplementedError(f"Not implemented for adapter of type {adapter_type}")


    def get_up_project_key(model_type, layer_idx, adapter_type):
        if adapter_type == "pfeiffer":
            return f"{model_type}.encoder.layer.{layer_idx}.output.adapters.ner.adapter_up.weight", None
        elif adapter_type == "lora":
            return (f"{model_type}.encoder.layer.{layer_idx}.attention.self.query.loras.ner.lora_B",
                    f"{model_type}.encoder.layer.{layer_idx}.attention.self.value.loras.ner.lora_B")
        else:
            raise NotImplementedError(f"Not implemented for adapter of type {adapter_type}")


    def get_correlation_indices(teacher_embeddings, student_embeddings):
        correlation_matrix = np.zeros((student_embeddings.size(-1), teacher_embeddings.size(-1)))

        print("Doing correlation alignment on corr matrix size", correlation_matrix.shape)

        for i in range(student_embeddings.size(-1)):
            correlations = r_regression(teacher_embeddings.cpu(), student_embeddings[:, i].cpu())
            correlation_matrix[i] = correlations

        cost, mapping = linear_max_sum_assignment(correlation_matrix)
        print("Cost", cost)
        indices = [x[1] for x in sorted(mapping, key=lambda x: x[0])]
        return indices


    adapter = torch.load(os.path.join(args.adapter_path, "pytorch_adapter.bin"), map_location=torch.device('cpu'))
    for layer_idx in student_probing_model.before.keys():
        teacher_layer_idx = layer_idx * args.prune_every_k
        teacher_embeddings = torch.stack(teacher_probing_model.before[teacher_layer_idx])
        student_embeddings = torch.stack(student_probing_model.before[layer_idx])

        print("Correlation between embeddings of sizes:")
        print("Teacher:", teacher_embeddings.size())
        print("Student:", student_embeddings.size())

        indices = get_correlation_indices(
            teacher_embeddings, student_embeddings
        )

        if args.adapter_type == "pfeiffer":
            # DOWN PROJECTION
            down_projection_key, _ = get_down_project_key(args.teacher_model_type, teacher_layer_idx, args.adapter_type)

            down_projection = adapter[down_projection_key]
            pruned_down_projection = down_projection[:, indices]
            adapter[down_projection_key] = pruned_down_projection

            # UP PROJECTION
            up_projection_key, _ = get_up_project_key(args.teacher_model_type, teacher_layer_idx, args.adapter_type)

            up_projection = adapter[up_projection_key]
            pruned_up_projection = up_projection[indices, :]
            adapter[up_projection_key] = pruned_up_projection

            # UP PROJECTION BIAS
            up_bias_key = f"{args.teacher_model_type}.encoder.layer.{teacher_layer_idx}.output.adapters.ner.adapter_up.bias"

            up_bias = adapter[up_bias_key]
            pruned_up_bias = up_bias[indices]
            adapter[up_bias_key] = pruned_up_bias

        elif args.adapter_type == "lora":
            # 1. DOWN PROJECTION
            down_projection_key_query, down_projection_key_value = get_down_project_key(args.teacher_model_type,
                                                                                        teacher_layer_idx,
                                                                                        args.adapter_type)

            # 1.1 QUERY LORA
            down_projection = adapter[down_projection_key_query]
            pruned_down_projection = down_projection[:, indices]
            adapter[down_projection_key_query] = pruned_down_projection

            # 1.2 VALUE LORA
            down_projection = adapter[down_projection_key_value]
            pruned_down_projection = down_projection[:, indices]
            adapter[down_projection_key_value] = pruned_down_projection

            # 2. UP PROJECTION LORA
            up_projection_key_query, up_projection_key_value = get_up_project_key(args.teacher_model_type,
                                                                                  teacher_layer_idx, args.adapter_type)

            # 2.1 QUERY LORA
            up_projection = adapter[up_projection_key_query]
            pruned_up_projection = up_projection[indices, :]
            adapter[up_projection_key_query] = pruned_up_projection

            # 2.2 VALUE LORA
            up_projection = adapter[up_projection_key_value]
            pruned_up_projection = up_projection[indices, :]
            adapter[up_projection_key_value] = pruned_up_projection

        else:
            raise NotImplementedError(f"Not implemented for adapter of type {args.adapter_type}")

    os.makedirs(args.pruned_adapter_path, exist_ok=True)

    new_adapter = {}
    for k, v in adapter.items():
        new_adapter[f"_new.{k}"] = v

    adapter = new_adapter
    for k, v in adapter.items():
        print(k, v.size())

    torch.save(adapter, os.path.join(args.pruned_adapter_path, "pytorch_adapter.bin"))

    with open(os.path.join(args.adapter_path, "adapter_config.json")) as f:
        config = json.load(f)
        config["hidden_size"] = args.reduction_size

        if args.adapter_type == "pfeiffer":
            config["config"]["reduction_factor"] = args.student_reduction_factor

    with open(os.path.join(args.pruned_adapter_path, "adapter_config.json"), "w") as f:
        json.dump(config, f)
