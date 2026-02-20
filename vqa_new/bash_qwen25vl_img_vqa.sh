#!/bin/bash
set -e

# IMAGE_DIR="/proj/berzelius-2024-90/users/datasets/mmreid/Market-1501-v15.09.15"
IMAGE_DIR="/4tb/dataset/social_media/ins/24_1/media_files/2024-01-01_2024-03-01"  
# OUTPUT_DIR="/proj/berzelius-2024-90/users/x_liumi/Qwen2.5-VL/output/market1501_img_vqa"
OUTPUT_DIR="/4tb/dataset/social_media/ins/24_1/Qwen2.5-VL/output/version_4"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

BATCH_SIZE=16
NUM_SAMPLES=200000  # cap if needed; omit or set empty to use all
MAX_NEW_TOKENS=500
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=50
REPETITION_PENALTY=1.05
MIN_PIXELS=$((256*28*28))    # 256×28×28 = 200704
# MAX_PIXELS=$((1280*28*28))   # 1280×28×28 = 1003520
MAX_PIXELS=$((384*28*28))
DTYPE="bfloat16"         # auto|bfloat16|float16
FLASH_ATTN2=0            # 1 to enable
DEVICE_MAP="cuda"        # auto|cuda|cpu

QUESTIONS=(
  # "Does this image contain any visible text? Answer Yes or No. If Yes, extract all the text exactly as it appears."
  # "If the image contains text, provide a clean transcription of all visible text, preserving line breaks. If no text, answer 'No text found'."

  # "Does the image contain one or more humans? Answer Yes or No."
  # "If the image contains humans, describe each person’s perceived gender, approximate age band, outfit, pose or activity, and facial expression."
  # "If multiple people are present, describe their interactions, relative positions, and any notable group activity."
  # "If any person is recognizable (e.g., a celebrity or public figure), provide their likely name. If uncertain, answer 'Unclear'."

  # "Describe the image in one concise paragraph, focusing on key objects, scene type, and visual style."
  # "Classify the image into one of the following categories: person/people, text/graphic, landscape/nature, indoor scene, object/product, abstract/pattern, or other. Then justify briefly."

  # "Does the image contain any explicit, violent, hateful, or 18+ content? Answer with Yes/No and specify the category if Yes."
  # "Evaluate whether this image is safe-for-work (SFW) or not-safe-for-work (NSFW). Provide a short reason."

  # "What is the main theme or topic of this image? Choose from: politics, entertainment, sports, technology, daily life, advertisement, or other. Provide one sentence explanation."
  # "If the image is related to news or current events, summarize the likely subject matter in one sentence."

  # "Provide a balanced description of the image, including people, objects, text, and background elements."

  # "Does this image contain any visible text? Answer Yes or No. If Yes, transcribe all the text exactly as it appears." # OCR
  # "Does the image contain one or more humans? If Yes, describe each person’s name (if you know), gender, approximate age, outfit, activity, facial expression, and interactions if multiple." # Human
  # "Provide a concise description of the image, including key objects, background, and overall scene." # General
  # "Does the image contain explicit, offensive, violent, hateful, 18+, or any other inappropriate content? Answer Yes or No. If Yes, specify the category." # Toxicity
  # "What is the main theme of the image? Choose from: politics, entertainment, sports, technology, daily life, advertisement, abstract/pattern, or other. Provide a short justification." # Topic
  
  # v4
  "Does this image contain any visible text? Answer Yes or No. If Yes, transcribe the text exactly as it appears without paraphrasing or interpretation." # OCR
  "Does the image contain one or more humans? Answer Yes or No.\nIf Yes, for each visible person, provide a separate description including:\n- Person number (e.g., Person 1, Person 2, ...),\n- Name if recognizable,\n- Gender,\n- Approximate age,\n- Outfit,\n- Activity or pose,\n- Facial expression,\n- Interactions with other people (if any).\nKeep the description concise and factual, avoiding assumptions." # Human
  "Provide a concise description of the image in 2–3 sentences, mentioning the key objects, people, actions, background, texts, and overall scene." # General
  "Does the image contain violent, hateful, unsafe, 18+, or any other offensive content (e.g. alcohol, weapons, blood, cigarette, insulting gestures) or texts? Answer Yes or No. If Yes, specify the category clearly." # Toxicity
  "What is the main theme of the image? Choose the most specific category that fits best from: politics, entertainment, sports, technology, anime, gaming, fashion, beauty/makeup, photography, food, pets/animals, nature, travel, memes, advertising, daily life, art, health/fitness, science, religion, history, news/crime. Avoid 'other' unless nothing else fits. Provide a short justification." # Topic
  "If the image contains one or more humans, rate each visible person’s facial and/or body attractiveness on a scale of 1–10, where 1 = least attractive and 10 = most attractive, based on general visual aesthetics, symmetry, and presentation.\nIf no humans are visible, answer 'No human detected.'\nProvide your answer in the format:\nPerson 1: [score]; Person 2: [score]; ...\nOptionally include a one-sentence justification." # Attractiveness (1–10)
  "If the image contains one or more humans, classify each visible person’s facial and/or body attractiveness into one of three levels: 'Low', 'Medium', or 'High', based on general aesthetic impression, symmetry, and presentation.\nIf no humans are visible, answer 'No human detected.'\nProvide your answer in the format:\nPerson 1: [level]; Person 2: [level];\nOptionally include a short justification." # Attractiveness (Low/Med/High)
  "Rate the overall toxicity level of the image on a continuous scale from 0.0 to 1.0, where 0.0 means completely safe or non-toxic, and 1.0 means extremely toxic or unsafe.\nToxicity includes explicit, offensive, violent, hateful, unsafe, 18+, or otherwise inappropriate content such as alcohol, weapons, blood, cigarettes, or insulting gestures, as well as toxic text content.\nIf no such elements appear, give 0.0.\nProvide the numeric score and a short justification." # Toxicity (score)
)

python /home/labad/minxing/code/Qwen2.5-VL/vqa/qwen25vl_img_vqa.py \
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
