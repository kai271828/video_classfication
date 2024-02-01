## install

```
pip3 install -r requirements.txt
```

## example

```
python3 train.py \
    --train_dataset_dir ./dataset/train \
    --val_dataset_dir ./dataset/val \
    --sample_rate 4 \
    --model_name_or_path MCG-NJU/videomae-base-finetuned-kinetics \
    --output_dir videomae-base-fatigue-detection \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps 50 \
    --eval_steps 50 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --load_best_model_at_end True \
    --metric_for_best_model "accuracy" \
    --push_to_hub True \
    --num_train_epochs 1
```