# Is Modularity Transferable? A Case Study through the Lens of Knowledge Distillation
This project provides the codebase required to reproduce experiments introduced in the paper [Is Modularity Transferable? A Case Study through the Lens of Knowledge Distillation](https://aclanthology.org/2024.lrec-main.817.pdf) at [LREC-COLING 2024](https://lrec-coling-2024.org/).

## Citation
```
@inproceedings{klimaszewski-etal-2024-modularity-transferable,
    title = "Is Modularity Transferable? A Case Study through the Lens of Knowledge Distillation",
    author = "Klimaszewski, Mateusz  and
      Andruszkiewicz, Piotr  and
      Birch, Alexandra",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.817",
    pages = "9352--9360",
    abstract = "The rise of Modular Deep Learning showcases its potential in various Natural Language Processing applications. Parameter-efficient fine-tuning (PEFT) modularity has been shown to work for various use cases, from domain adaptation to multilingual setups. However, all this work covers the case where the modular components are trained and deployed within one single Pre-trained Language Model (PLM). This model-specific setup is a substantial limitation on the very modularity that modular architectures are trying to achieve. We ask whether current modular approaches are transferable between models and whether we can transfer the modules from more robust and larger PLMs to smaller ones. In this work, we aim to fill this gap via a lens of Knowledge Distillation, commonly used for model compression, and present an extremely straightforward approach to transferring pre-trained, task-specific PEFT modules between same-family PLMs. Moreover, we propose a method that allows the transfer of modules between incompatible PLMs without any change in the inference complexity. The experiments on Named Entity Recognition, Natural Language Inference, and Paraphrase Identification tasks over multiple languages and PEFT methods showcase the initial potential of transferable modularity.",
}
```
## Installation

```bash
conda create -n mKD python=3.9
conda activate mKD

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# or
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

pip install adapter-transformers==3.2.1 wandb seqeval datasets evaluate scipy scikit-learn
```

## Experiments
Remember to set env variables:
```bash
# Example
DATASET="Babelscape/wikineural"
LANGUAGE="en"
ADAPTER_TYPE="pfeiffer"
ADAPTER_CONFIG="$ADAPTER_TYPE[r=8]"
OUTPUT_DIR="/tmp/results"
TEACHER_MODEL_TYPE="roberta"
STUDENT_MODEL_TYPE="roberta"
BATCH_SIZE=64
TOPK=100
EPOCHS=10
BATCH_SIZE=16
WORKERS=2
LR=1e-5
TEACHER_MODEL="xlm-roberta-large"
STUDENT_MODEL="xlm-roberta-base"
SKIP=2
```
### Teacher training

```bash
python3 run_ner.py \
  --model_name_or_path $TEACHER_MODEL \
  --dataset_name $DATASET \
  --dataset_config_name $LANGUAGE \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --load_best_model_at_end \
  --evaluation_strategy steps \
  --metric_for_best_model f1 \
  --save_total_limit 1 \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --fp16 \
  --do_train \
  --do_eval \
  --do_predict \
  --run_name $WANDB_NAME \
  --train_adapter --adapter_config "$ADAPTER_CONFIG"
```
### Prunning & Alignment (optional)
```bash
python3 ../../src/mkd/prune_and_align.py \
    --teacher_model_type $TEACHER_MODEL_TYPE \
    --teacher_model_name $TEACHER_MODEL \
    --student_model_type $STUDENT_MODEL_TYPE \
    --student_model_name $STUDENT_MODEL \
    --adapter_type $ADAPTER_TYPE \
    --dataset_name $DATASET \
    --dataset_config_name $LANGUAGE \
    --adapter_path "$OUTPUT_DIR/ner" \
    --pruned_adapter_path "$OUTPUT_DIR/pruned_ner_${ADAPTER_TYPE}_${STUDENT_MODEL}" \
    --topk $TOPK \
    --prune_every_k $SKIP
```

### Student training (SKIP, incompatible version)
```bash
python3 run_ner.py \
  --model_name_or_path $STUDENT_MODEL \
  --dataset_name $DATASET \
  --dataset_config_name $LANGUAGE \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --load_best_model_at_end \
  --evaluation_strategy steps \
  --metric_for_best_model f1 \
  --save_total_limit 1 \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --fp16 \
  --do_train \
  --do_eval \
  --do_predict \
  --train_adapter --adapter_config "$ADAPTER_CONFIG" \
  --load_adapter "$TEACHER_OUTPUT_DIR/pruned_ner_${ADAPTER_TYPE}_${STUDENT_MODEL}/" \
  --load_every_k_layer $SKIP \
  --teacher_type $TEACHER_MODEL_TYPE \
  --student_type $STUDENT_MODEL_TYPE
```