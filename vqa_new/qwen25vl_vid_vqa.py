#!/usr/bin/env python3
import argparse, os, csv
from typing import List, Optional
from tqdm import tqdm
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # handles frame extraction / resizing per message

VIDEO_EXTS = ("mp4","mov","avi","mkv","webm","m4v")

def find_videos(root: str) -> List[str]:
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                paths.append(os.path.join(r, f))
    paths.sort()
    return paths

def main():
    p = argparse.ArgumentParser(description="Qwen2.5-VL Video VQA")
    p.add_argument("--video_dir", required=True, help="Directory with videos (recursively).")
    p.add_argument("--output_dir", default="qwen25vl_vid_vqa", help="Where to save results.")
    p.add_argument("--questions", nargs="+", default=["Describe the video."], help="List of questions.")
    p.add_argument("--num_samples", type=int, default=None, help="Cap number of videos.")
    # sampling / resize knobs
    p.add_argument("--fps", type=float, default=2.0, help="FPS to sample (qwen-vl-utils).")
    p.add_argument("--nframes", type=int, default=None, help="Alternative to fps; multiple of 2.")
    p.add_argument("--total_pixels", type=int, default=20480*28*28, help="Global pixel budget for video tokens.")
    p.add_argument("--min_pixels", type=int, default=16*28*28, help="Per-frame lower pixel bound.")
    p.add_argument("--max_pixels", type=int, default=None, help="Optional per-frame upper pixel bound.")
    # model/gen
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--flash_attn2", action="store_true")
    p.add_argument("--device_map", default="auto")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = "auto"
    if args.dtype.lower() in ["bfloat16", "bf16"]:
        dtype = torch.bfloat16
    elif args.dtype.lower() in ["float16", "fp16", "half"]:
        dtype = torch.float16

    attn_impl = "flash_attention_2" if args.flash_attn2 else "sdpa"
    processor = AutoProcessor.from_pretrained(args.model_name)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map=args.device_map,
    )
    device = model.device
    print(f"Loaded model on {device}; dtype={model.dtype}; flash_attn2={args.flash_attn2}")

    vids = find_videos(args.video_dir)
    if args.num_samples is not None:
        vids = vids[:args.num_samples]
    print(f"Total videos: {len(vids)}")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    for q_idx, question in enumerate(args.questions, start=1):
        print(f"\n[Q{q_idx}] {question}")
        q_out_dir = os.path.join(args.output_dir, f"question_{q_idx}")
        os.makedirs(q_out_dir, exist_ok=True)
        csv_path = os.path.join(q_out_dir, "results.csv")
        empty_log = os.path.join(q_out_dir, "empty_answers.log")

        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, \
             open(empty_log, "w", encoding="utf-8") as flog:

            writer = csv.DictWriter(fcsv, fieldnames=["filename", "answer"])
            writer.writeheader()
            flog.write("Empty Answers Log\n=================\n")

            for vpath in tqdm(vids, desc=f"Q{q_idx}"):
                # Build one-sample message (video settings on the content dict)
                content_video = {
                    "type": "video",
                    "video": f"file://{vpath}",   # local path -> file:// scheme
                    "total_pixels": args.total_pixels,
                    "min_pixels": args.min_pixels
                }
                if args.max_pixels is not None:
                    content_video["max_pixels"] = args.max_pixels
                if args.nframes is not None:
                    content_video["nframes"] = args.nframes
                else:
                    content_video["fps"] = args.fps

                messages = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": question},
                         content_video
                     ]}
                ]

                # Prepare inputs (qwen-vl-utils will read & resize frames)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)

                try:
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        fps=video_kwargs.get("fps", None),
                        padding=True,
                        return_tensors="pt",
                    ).to(device)

                    with torch.no_grad():
                        generated = model.generate(**inputs, **gen_kwargs)

                    trimmed = [out[len(inp):] for inp,out in zip(inputs.input_ids,generated)]
                    ans = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    ans = (ans or "").strip().replace("\n"," ")
                    if ans.startswith("addCriterion"):
                        ans = ans.replace("addCriterion","",1).strip()
                    if not ans:
                        flog.write(f"Empty answer for video: {vpath}\n")
                        ans = "No answer generated."
                except Exception as e:
                    flog.write(f"Generation error for video: {vpath} :: {e}\n")
                    ans = "Error"

                writer.writerow({"filename": vpath, "answer": ans})

        print(f"[Saved] {csv_path}")

    print("All questions processed.")

if __name__ == "__main__":
    main()
