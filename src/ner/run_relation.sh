PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-bert-base-uncased}
OUTPUT_DIR=${OUTPUT_DIR:-./result/run_relation}
TRAIN_FILE=${TRAIN_FILE:-./data/data_split/train.json}
ENTITY_OUTPUT_DIR=${ENTITY_OUTPUT_DIR:-./result/run_entity}
ENTITY_PRED_DEV=${ENTITY_PRED_DEV:-ent_pred_dev.json}
ENTITY_PRED_TEST=${ENTITY_PRED_TEST:-ent_pred_test.json}

"$PYTHON_BIN" run_relation.py \
    --task mimic01 \
    --do_train \
    --do_eval \
    --eval_with_gold \
    --model "$MODEL" \
    --do_lower_case \
    --train_file "$TRAIN_FILE" \
    --entity_output_dir "$ENTITY_OUTPUT_DIR" \
    --entity_predictions_dev "$ENTITY_PRED_DEV" \
    --entity_predictions_test "$ENTITY_PRED_TEST" \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --context_window 100 \
    --max_seq_length 256 \
    --no_cuda \
    --output_dir "$OUTPUT_DIR"
