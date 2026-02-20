#!/usr/bin/env python3
import argparse
import os
import csv
import random
from typing import List
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def setup_distributed():
    """Initialize torch.distributed if not yet initialized."""
    if not dist.is_initialized():
        # Expect environment variables from torchrun (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK)
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))  # torchrun always sets LOCAL_RANK
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def find_files(root: str, exts: List[str]) -> List[str]:
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(tuple(exts)):
                paths.append(os.path.join(r, f))
    paths.sort()
    return paths


def partition_indices(total_size: int, world_size: int, rank: int) -> List[int]:
    """Contiguous partition so each rank gets a unique slice."""
    per_rank = total_size // world_size
    rem = total_size % world_size
    if rank < rem:
        start = rank * (per_rank + 1)
        end = start + (per_rank + 1)
    else:
        start = rank * per_rank + rem
        end = start + per_rank
    return list(range(start, end))


def load_images(paths: List[str]) -> (List[Image.Image], List[bool]):
    imgs, mask = [], []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
            mask.append(True)
        except Exception as e:
            # Fallback white image, mark invalid
            print(f"[WARN] Failed to open {p}: {e}")
            imgs.append(Image.new("RGB", (224, 224), (255, 255, 255)))
            mask.append(False)
    return imgs, mask


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Image VQA (multi-GPU via torchrun)")
    parser.add_argument("--image_dir", required=True, help="Root dir of images (recursive).")
    parser.add_argument("--output_dir", default="qwen25vl_img_vqa_ddp", help="Output root dir.")
    parser.add_argument("--questions", nargs="+", default=["Describe the image."], help="Questions to ask.")
    parser.add_argument("--num_samples", type=int, default=None, help="Global cap on images (after sort/shuffle).")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="(Reserved) Not used; PIL loading inline.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle list before applying --num_samples.")
    parser.add_argument("--seed", type=int, default=2024, help="Shuffle seed.")

    # model/inference knobs
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id or local path.")
    parser.add_argument("--dtype", default="auto", help="auto|bfloat16|float16|float32")
    parser.add_argument("--flash_attn2", action="store_true", help="Enable Flash-Attention 2 (if installed).")

    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # vision token budget (optional, only if you want to clamp pixel range)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)

    args = parser.parse_args()

    # ----- DDP setup -----
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ----- Seed & output -----
    if args.shuffle:
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Build global image list (identical on all ranks) -----
    image_paths = find_files(args.image_dir, list(IMG_EXTS))
    if args.shuffle:
        random.shuffle(image_paths)
    if args.num_samples is not None:
        image_paths = image_paths[:args.num_samples]

    total_images = len(image_paths)
    if rank == 0:
        print(f"[DDP] world_size={world_size} | total_images={total_images} | per-gpu batch={args.batch_size}")

    # Partition across ranks
    my_indices = partition_indices(total_images, world_size, rank)
    my_paths = [image_paths[i] for i in my_indices]
    print(f"[Rank {rank}] assigned {len(my_paths)} images")

    # ----- Load model/processor (1 copy per rank) -----
    # dtype
    dtype = "auto"
    if args.dtype.lower() in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    elif args.dtype.lower() in ("float16", "fp16", "half"):
        dtype = torch.float16
    elif args.dtype.lower() in ("float32", "fp32"):
        dtype = torch.float32

    attn_impl = "flash_attention_2" if args.flash_attn2 else "sdpa"

    # optional pixel bounds for processor
    proc_kwargs = {}
    if args.min_pixels is not None: proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None: proc_kwargs["max_pixels"] = args.max_pixels

    processor = AutoProcessor.from_pretrained(args.model_name, **proc_kwargs)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map=None,         # load on CPU then move
    )
    model = model.to(device)
    model.eval()
    if rank == 0:
        print(f"[Model] loaded {args.model_name} on {device} | dtype={model.dtype} | attn={attn_impl}")

    # ----- Generation params -----
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    # ----- Loop over questions -----
    for q_idx, question in enumerate(args.questions, start=1):
        # Per-question output dir
        q_dir = os.path.join(args.output_dir, f"question_{q_idx}")
        os.makedirs(q_dir, exist_ok=True)

        # Per-rank interim files
        rank_csv = os.path.join(q_dir, f"rank_{rank}.csv")
        rank_log = os.path.join(q_dir, f"rank_{rank}.empty_answers.log")

        # Process this rank's images in batches
        with open(rank_csv, "w", newline="", encoding="utf-8") as fcsv, \
             open(rank_log, "w", encoding="utf-8") as flog:

            writer = csv.DictWriter(fcsv, fieldnames=["filename", "answer"])
            writer.writeheader()
            flog.write("Empty Answers Log\n=================\n")

            if rank == 0:
                iterator = tqdm(range(0, len(my_paths), args.batch_size), desc=f"Q{q_idx}")
            else:
                iterator = range(0, len(my_paths), args.batch_size)

            for i in iterator:
                batch_paths = my_paths[i:i+args.batch_size]
                batch_imgs, valid_mask = load_images(batch_paths)

                # Build chat messages containing actual file URIs (the processor uses images=... for pixel data)
                messages = []
                for p in batch_paths:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{p}"},
                            {"type": "text", "text": question},
                        ],
                    })

                # Build per-sample chat text
                texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
                         for m in messages]

                # Pack for the model
                inputs = processor(
                    text=texts,
                    images=batch_imgs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)

                # Keep only newly generated tokens
                trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
                answers = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                # Write rows
                for p, ans, ok in zip(batch_paths, answers, valid_mask):
                    ans = (ans or "").strip().replace("\n", " ")
                    if not ans:
                        flog.write(f"Empty answer for image: {p}\n")
                        ans = "No answer generated."
                    if not ok:
                        ans = f"[LOAD_ERROR] {ans}"
                    writer.writerow({"filename": p, "answer": ans})

        # ----- Merge per-rank files on rank 0 -----
        dist.barrier()
        if rank == 0:
            final_csv = os.path.join(q_dir, "results.csv")
            final_log = os.path.join(q_dir, "empty_answers.log")
            # merge CSVs
            with open(final_csv, "w", newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(fout, fieldnames=["filename", "answer"])
                writer.writeheader()
                for r in range(world_size):
                    part = os.path.join(q_dir, f"rank_{r}.csv")
                    if not os.path.exists(part):
                        continue
                    with open(part, "r", encoding="utf-8") as fin:
                        # skip header
                        next(fin, None)
                        for line in fin:
                            fout.write(line)
            # merge logs
            with open(final_log, "w", encoding="utf-8") as lf:
                lf.write("Empty Answers Log\n=================\n")
                for r in range(world_size):
                    part = os.path.join(q_dir, f"rank_{r}.empty_answers.log")
                    if not os.path.exists(part):
                        continue
                    with open(part, "r", encoding="utf-8") as pf:
                        # skip header lines in parts
                        lines = pf.readlines()
                        # keep content after header (2 lines)
                        lf.writelines(lines[2:] if len(lines) > 2 else [])

            # cleanup parts
            for r in range(world_size):
                for nm in (f"rank_{r}.csv", f"rank_{r}.empty_answers.log"):
                    pth = os.path.join(q_dir, nm)
                    if os.path.exists(pth):
                        try:
                            os.remove(pth)
                        except:
                            pass

            print(f"[Q{q_idx}] merged -> {final_csv}")

        dist.barrier()

    cleanup_distributed()
    if rank == 0:
        print("All questions processed.")


if __name__ == "__main__":
    main()
