#!/bin/bash
set -e

IMAGE_DIR="/proj/berzelius-2024-90/users/datasets/mmreid/Market-1501-v15.09.15"
OUTPUT_DIR="/proj/berzelius-2024-90/users/x_liumi/Qwen2.5-VL/output/market1501_img_vqa"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

BATCH_SIZE=64
NUM_SAMPLES=200000  # cap if needed; omit or set empty to use all
MAX_NEW_TOKENS=200
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=50
REPETITION_PENALTY=1.05
MIN_PIXELS=$((256*28*28))    # 256×28×28 = 200704
MAX_PIXELS=$((1280*28*28))   # 1280×28×28 = 1003520
DTYPE="bfloat16"         # auto|bfloat16|float16
FLASH_ATTN2=0            # 1 to enable
DEVICE_MAP="cuda"        # auto|cuda|cpu

QUESTIONS=(
  "Describe the image."
)

python qwen25vl_img_vqa.py \
  --image_dir "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --top_k $TOP_K \
  --repetition_penalty $REPETITION_PENALTY \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  ${MIN_PIXELS:+--min_pixels $MIN_PIXELS} \
  ${MAX_PIXELS:+--max_pixels $MAX_PIXELS} \
  $( [[ -n "$NUM_SAMPLES" ]] && echo --num_samples $NUM_SAMPLES ) \
  $( [[ "$FLASH_ATTN2" == "1" ]] && echo --flash_attn2 ) \
  --questions "${QUESTIONS[@]}"
