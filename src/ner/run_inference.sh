PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-bert-base-uncased}
TEST_DATA=${TEST_DATA:-./data/chexpert_plus_groundtruth.json}
ENTITY_OUTPUT_DIR=${ENTITY_OUTPUT_DIR:-./result/run_entity}
RELATION_OUTPUT_DIR=${RELATION_OUTPUT_DIR:-./result/run_relation}
ENTITY_TEST_PRED=${ENTITY_TEST_PRED:-ent_pred_test.json}
RELATION_TEST_PRED=${RELATION_TEST_PRED:-ent_rel_pred_test.json}

"$PYTHON_BIN" run_entity.py \
    --do_eval \
    --eval_test \
    --learning_rate=1e-5 \
    --task_learning_rate=5e-4 \
    --train_batch_size=8 \
    --eval_batch_size 256 \
    --context_window 100 \
    --task mimic01 \
    --data_dir ./data \
    --test_data "$TEST_DATA" \
    --test_pred_filename "$ENTITY_TEST_PRED" \
    --model "$MODEL" \
    --output_dir "$ENTITY_OUTPUT_DIR"

"$PYTHON_BIN" run_relation.py \
    --task mimic01 \
    --do_eval \
    --eval_test \
    --model "$MODEL" \
    --do_lower_case \
    --train_batch_size 16 \
    --eval_batch_size 256 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --context_window 100 \
    --max_seq_length 256 \
    --no_cuda \
    --entity_output_dir "$ENTITY_OUTPUT_DIR" \
    --entity_predictions_test "$ENTITY_TEST_PRED" \
    --output_dir "$RELATION_OUTPUT_DIR" \
    --prediction_file "$RELATION_TEST_PRED"
