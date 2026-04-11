#!/bin/bash
# Collect routing data for training the expert routing predictor.
#
# Usage:
#   ./scripts/collect_routing_data.sh [n_prompts]
#
# Prerequisites:
#   pip install datasets
#
# This script will:
#   1. Generate a balanced prompt dataset
#   2. Start the server with routing data collection enabled
#   3. Feed all prompts to the server
#   4. Train the routing predictor
#   5. Output routing_predictor.bin ready for use

set -e

N_PROMPTS="${1:-200}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$BUILD_DIR/routing_training"

MODEL="/mnt/sdc1/Downloads4/hf/gemma-4-26B-A4B-it-UD-IQ2_M.gguf"
CHAT_TEMPLATE="/mnt/sdc1/Downloads4/google-gemma-4-31B-it-interleaved.jinja"
SERVER="$BUILD_DIR/build/bin/llama-server"
PORT=8090  # use non-default port to avoid conflicts

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "=== Step 1: Prepare prompts ==="
python3 "$SCRIPT_DIR/prepare_routing_dataset.py" \
    -o "$WORK_DIR/prompts.jsonl" \
    -n "$N_PROMPTS"

TOTAL=$(wc -l < "$WORK_DIR/prompts.jsonl")
if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No prompts generated. Install datasets: pip install datasets"
    exit 1
fi
echo "Generated $TOTAL prompts"

echo ""
echo "=== Step 2: Start server with routing collector ==="
$SERVER \
    -m "$MODEL" \
    -ngl 99 -fa on -c 8192 -t 20 \
    --ring-experts 4096 \
    --routing-collector "$WORK_DIR/routing_data.bin" \
    --chat-template-file "$CHAT_TEMPLATE" --jinja \
    --reasoning-budget 0 \
    --override-kv gemma4.final_logit_softcapping=float:25 \
    --port $PORT \
    --log-disable \
    &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server (pid $SERVER_PID) to start..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server exited unexpectedly"
        exit 1
    fi
    sleep 2
done

echo ""
echo "=== Step 3: Feed prompts (this takes a while) ==="
COUNT=0
while IFS= read -r line; do
    COUNT=$((COUNT + 1))
    curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$line" > /dev/null 2>&1 || true
    printf "\r  Progress: %d/%d" "$COUNT" "$TOTAL"
done < "$WORK_DIR/prompts.jsonl"
echo ""
echo "Done feeding $COUNT prompts."

echo ""
echo "=== Step 4: Stop server ==="
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 1

if [ ! -f "$WORK_DIR/routing_data.bin" ]; then
    echo "ERROR: routing_data.bin not created. Server may not have collected data."
    exit 1
fi

DATA_SIZE=$(stat -c%s "$WORK_DIR/routing_data.bin" 2>/dev/null || stat -f%z "$WORK_DIR/routing_data.bin")
echo "Collected $(($DATA_SIZE / 1024))KB of routing data"

echo ""
echo "=== Step 5: Train predictor ==="
python3 "$SCRIPT_DIR/train_routing_predictor_hidden.py" \
    --data "$WORK_DIR/routing_data.bin" \
    --output "$WORK_DIR/routing_predictor.bin" \
    --lookahead 3 \
    --epochs 20

echo ""
echo "=== Done! ==="
echo ""
echo "To use the predictor, add this flag to your server command:"
echo "  --routing-predictor $WORK_DIR/routing_predictor.bin"
