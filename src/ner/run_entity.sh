PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-bert-base-uncased}
OUTPUT_DIR=${OUTPUT_DIR:-./result/run_entity}
TRAIN_DATA=${TRAIN_DATA:-./data/data_split/train.json}
DEV_DATA=${DEV_DATA:-./data/data_split/test.json}
TEST_DATA=${TEST_DATA:-./data/data_split/test.json}

"$PYTHON_BIN" run_entity.py \
    --do_train \
    --do_eval \
    --eval_test \
    --learning_rate=1e-5 \
    --task_learning_rate=5e-4 \
    --train_batch_size=8 \
    --eval_batch_size=64 \
    --num_epoch=1 \
    --context_window 100 \
    --task mimic01 \
    --data_dir ./data \
    --train_data "$TRAIN_DATA" \
    --dev_data "$DEV_DATA" \
    --test_data "$TEST_DATA" \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR"

