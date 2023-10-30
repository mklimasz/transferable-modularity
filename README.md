# Is Modularity Transferable? A Case Study through the Lens of Knowledge Distillation
This project provides the codebase required to reproduce experiments introduced in the paper [Is Modularity Transferable? A Case Study through the Lens of Knowledge Distillation](https://arxiv.org/pdf/2403.18804) at [LREC-COLING 2024](https://lrec-coling-2024.org/).

## Citation
```
@misc{klimaszewski2024modularity,
      title={Is Modularity Transferable? A Case Study through the Lens of Knowledge Distillation}, 
      author={Mateusz Klimaszewski and Piotr Andruszkiewicz and Alexandra Birch},
      year={2024},
      eprint={2403.18804},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
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