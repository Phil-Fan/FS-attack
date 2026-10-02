#!/bin/bash
set -u

# Default parameters
MODEL_TYPE=${MODEL_TYPE:-"all"}      # all, minirocket, inceptiontime, lightgbm
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-64}

echo "=========================================="
echo "Batch training comparison script (Probe CLI)"
echo "Model Type: $MODEL_TYPE"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Check whether RUN_OUTPUT_DIR is set
if [ -z "${RUN_OUTPUT_DIR:-}" ]; then
    echo "Error: RUN_OUTPUT_DIR environment variable must be set"
    echo "Example: RUN_OUTPUT_DIR=/path/to/datasets ./train.sh"
    exit 1
fi

if [ ! -d "$RUN_OUTPUT_DIR" ]; then
    echo "Error: directory does not exist: $RUN_OUTPUT_DIR"
    exit 1
fi

# Find all dataset directories (excluding the root, hidden dirs, and dirs starting with _)
mapfile -d '' DATASETS < <(
    find "$RUN_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d \
        ! -name '.*' ! -name '_*' -print0 | sort -z
)

if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "Error: no dataset directories found under $RUN_OUTPUT_DIR"
    exit 1
fi

echo "Found datasets:"
for d in "${DATASETS[@]}"; do
    echo "  - $(basename "$d")"
done
echo ""

# Summary results file
SUMMARY_FILE="$RUN_OUTPUT_DIR/training_summary.csv"
echo "dataset,model,accuracy,f1_macro,train_time_seconds" > "$SUMMARY_FILE"

# Temporary results directory
TEMP_DIR="$RUN_OUTPUT_DIR/.training_temp"
mkdir -p "$TEMP_DIR"

# Process each dataset
for DATASET_DIR in "${DATASETS[@]}"; do
    DATASET_NAME=$(basename "$DATASET_DIR")
    echo "=========================================="
    echo "Processing dataset: $DATASET_NAME"
    echo "=========================================="

    # Data directory
    DATA_DIR="$DATASET_DIR/data"
    OUTPUT_DIR="$DATASET_DIR/training"

    if [ ! -d "$DATA_DIR" ]; then
        echo "Skipping: data directory does not exist: $DATA_DIR"
        continue
    fi

    # Determine the list of models to train
    if [ "$MODEL_TYPE" = "all" ]; then
        MODELS="lightgbm"
    else
        MODELS="$MODEL_TYPE"
    fi

    # Train each model
    for MODEL in $MODELS; do
        echo ""
        echo ">>> Training $MODEL ..."

        # Run training
        if ! python probe/cli.py train "$DATA_DIR" \
            --model "$MODEL" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --output "$OUTPUT_DIR" \
            --device auto; then
            echo "    Failed: $MODEL"
            continue
        fi

        # Extract metrics
        METRICS_FILE="$OUTPUT_DIR/model_$MODEL/config.json"
        if [ -f "$METRICS_FILE" ]; then
            ACCURACY=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["val_accuracy"])' "$METRICS_FILE" 2>/dev/null || echo "0")
            F1=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["metrics"]["f1_macro"])' "$METRICS_FILE" 2>/dev/null || echo "0")

            # TrainTask stores the training duration in training_summary.json
            SUMMARY_JSON="$OUTPUT_DIR/training_summary.json"
            TRAIN_TIME=$(python3 -c 'import json, sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2] + "/train_time"])' "$SUMMARY_JSON" "$MODEL" 2>/dev/null || echo "0")

            echo "$DATASET_NAME,$MODEL,$ACCURACY,$F1,$TRAIN_TIME" >> "$SUMMARY_FILE"
        else
            echo "    Warning: metrics file not found: $METRICS_FILE"
        fi

        echo "    Done: $MODEL"
    done

    echo ""
done

# Clean up temporary directory
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

# Show summary results
echo ""
echo "Training results summary:"
echo "------------------------------------------"
cat "$SUMMARY_FILE"
echo ""

# Generate visualizations
echo ""
echo "Generating visualization comparison charts..."
python probe/visualize_results.py "$RUN_OUTPUT_DIR" --output "$RUN_OUTPUT_DIR/visualization" 2>/dev/null || echo "Visualization skipped (matplotlib unavailable)"

echo ""
echo "Results saved to:"
echo "  - Summary table: $SUMMARY_FILE"
echo "  - Visualizations: $RUN_OUTPUT_DIR/visualization/"
