#!/bin/bash
set -e

# ===== Adjust these =====
IMAGE_DIR="/proj/berzelius-2024-90/users/datasets/mmreid/Market-1501-v15.09.15"   # parent dir; script scans subfolders
OUTPUT_DIR="/proj/berzelius-2024-90/users/x_liumi/Qwen2.5-VL/output/market1501_img_vqa_ddp_cleaned"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

NUM_GPUS=8         # per node
NNODES=1
RDZV_ENDPOINT="localhost:29500"  # change port if occupied

BATCH_SIZE=64      # per GPU
NUM_SAMPLES=200000  # global cap; leave empty to use all
SHUFFLE=1
SEED=2024

# Token budget (optional). Use arithmetic expansion for ints.
MIN_PIXELS=$((256*28*28))      # 200704
MAX_PIXELS=$((1280*28*28))     # 1003520

# Generation
MAX_NEW_TOKENS=200
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=50
REPETITION_PENALTY=1.05

# Precision / attention impl
DTYPE="bfloat16"   # auto|bfloat16|float16|float32
FLASH_ATTN2=0      # set to 1 only if you have a compatible flash-attn installed

# Questions
QUESTIONS=(
#   "In one concise sentence, describe only the person’s stable physical traits: perceived gender, broad age group, height category, body build, limb/torso proportions, posture or gait, head/face shape impression, hair length and color, and any visible permanent marks (e.g., scars, tattoos). Avoid clothing, logos, and accessories."
#   "In 1–2 sentences, give a balanced description of the person that prioritizes body build, approximate age, height category, hair, notable physical features, and overall pose or motion. Mention clothing briefly only if it helps identification."
    "One compact sentence (≤25 words) describing only stable physical traits—perceived sex; broad age band; height category; build; limb–torso proportion; head/face shape; hair length & color; posture/gait; and any distinctive marks with location. Use concrete, discriminative words; omit unknown traits. Do not mention clothing/logos/accessories or start with “The person/individual,” and avoid the phrases “appears,” “likely,” “average height/build,” “proportionate,” and “No visible…”. Return only the sentence."
    "In ≤2 sentences, prioritize discriminative traits: (1) body build + approximate age band + height category—be specific (avoid “average,” “appears/likely”); (2) hair details + notable anatomical features (e.g., jawline, cheekbones, brow, facial hair, skin marks with location) + posture/gait/gesture. Mention clothing only if uniquely identifying, in ≤5 words; otherwise omit. Vary phrasing; do not start with “The person/individual”; avoid boilerplate. Output only the description."
)

# ===== Run =====

# Ensure we’re in the same directory as qwen25vl_img_vqa.py or pass absolute path
SCRIPT="qwen25vl_img_vqa_parallel.py"

# Build optional flags
PIXEL_FLAGS=()
[[ -n "$MIN_PIXELS" ]] && PIXEL_FLAGS+=( --min_pixels "$MIN_PIXELS" )
[[ -n "$MAX_PIXELS" ]] && PIXEL_FLAGS+=( --max_pixels "$MAX_PIXELS" )

SHUFFLE_FLAG=()
[[ "$SHUFFLE" == "1" ]] && SHUFFLE_FLAG+=( --shuffle )

NUM_SAMPLES_FLAG=()
[[ -n "$NUM_SAMPLES" ]] && NUM_SAMPLES_FLAG+=( --num_samples "$NUM_SAMPLES" )

FA2_FLAG=()
[[ "$FLASH_ATTN2" == "1" ]] && FA2_FLAG+=( --flash_attn2 )

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes="$NNODES" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$RDZV_ENDPOINT" \
  "$SCRIPT" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --seed "$SEED" \
    "${PIXEL_FLAGS[@]}" \
    "${SHUFFLE_FLAG[@]}" \
    "${NUM_SAMPLES_FLAG[@]}" \
    "${FA2_FLAG[@]}" \
    --questions "${QUESTIONS[@]}"
