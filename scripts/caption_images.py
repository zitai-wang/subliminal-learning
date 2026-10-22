#!/usr/bin/env python3
"""Caption MathVision/MathVista testmini images with a Claude vision model.

Produces one JSONL record per image: {"<id_field>": <id>, "caption": <text>}.
Captions describe only visual content (no solving, no answer leakage), so the
text can be paired with the question and fed to a text-only model for a
matched-information comparison against the VLM-with-image setting.
"""
import argparse
import base64
import glob
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Dataset, load_dataset
from PIL import Image

PROJECT_ROOT = "/data1/wzy/sub-learn"

CAPTION_PROMPT = (
    "You are given a single image accompanying a math problem (it may be a "
    "diagram, geometric figure, chart, graph, table, or a photographed scene). "
    "Describe ONLY the visual content needed to reason about the problem: "
    "all visible numbers, labels, units, and measurements; the type, count, and "
    "arrangement of shapes/objects; axis scales and plotted data values if it is "
    "a chart or graph; relative positions, sizes, and angles; any text or symbols "
    "shown. Be thorough and precise about quantities, since a reader must be able "
    "to solve the problem from your description alone.\n\n"
    "Do NOT solve the problem. Do NOT state, hint at, or imply any final answer. "
    "Do NOT add reasoning or commentary. Output only the objective visual "
    "description, as 3-8 sentences of plain text."
)


def load_env_file(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def load_mathvision_testmini():
    candidates = sorted(glob.glob(
        os.path.join(PROJECT_ROOT, ".hf_cache/datasets/math_vision/default/0.0.0/*/math_vision-testmini.arrow")
    ))
    if not candidates:
        raise FileNotFoundError("No cached MathVision testmini arrow file found under .hf_cache/datasets/math_vision")
    ds = Dataset.from_file(candidates[0])
    return ds, "id"


def load_mathvista_testmini():
    path = os.path.join(
        PROJECT_ROOT, "mathvista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
    )
    ds = load_dataset("parquet", data_files={"split": path})["split"]
    return ds, "pid"


DATASET_LOADERS = {
    "mathvision": load_mathvision_testmini,
    "mathvista": load_mathvista_testmini,
}


MAX_IMAGE_EDGE = 1568  # Claude downsamples beyond this anyway; also keeps payload small
MAX_B64_BYTES = 5 * 1024 * 1024


def image_to_png_b64(image: Image.Image) -> str:
    img = image.convert("RGB")
    if max(img.size) > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    if len(base64.b64encode(data)) > MAX_B64_BYTES:
        for quality in (90, 80, 70, 60, 50):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            if len(base64.b64encode(data)) <= MAX_B64_BYTES:
                return base64.b64encode(data).decode("utf-8"), "image/jpeg"
        # last resort: shrink further
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        data = buf.getvalue()
        return base64.b64encode(data).decode("utf-8"), "image/jpeg"

    return base64.b64encode(data).decode("utf-8"), "image/png"


def read_done_ids(output_path):
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(str(rec.get("_id")))
    return done


def caption_one(client, model, item_id, image, max_retries=5):
    b64, media_type = image_to_png_b64(image)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }],
            )
            text = resp.content[0].text.strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Failed to caption {item_id} after {max_retries} retries: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit; use small N for a dry run")
    args = parser.parse_args()

    load_env_file(os.path.join(PROJECT_ROOT, ".env"))

    import anthropic
    client = anthropic.Anthropic()

    ds, id_field = DATASET_LOADERS[args.dataset]()
    output_path = args.output or os.path.join(
        PROJECT_ROOT, "results/captions", f"{args.dataset}_captions.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    done_ids = read_done_ids(output_path)
    indices = [i for i in range(len(ds)) if str(ds[i][id_field]) not in done_ids]
    if args.limit:
        indices = indices[: args.limit]

    print(f"[{args.dataset}] total={len(ds)} already_done={len(done_ids)} to_process={len(indices)} -> {output_path}")

    write_lock = threading.Lock()

    def worker(idx):
        item = ds[idx]
        item_id = str(item[id_field])
        image = item["decoded_image"]
        caption = caption_one(client, args.model, item_id, image)
        record = {"_id": item_id, id_field: item[id_field], "caption": caption}
        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return item_id

    completed, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, idx): idx for idx in indices}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                fut.result()
                completed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[{args.dataset}] FAILED idx={idx}: {exc}")
            if (completed + failed) % 20 == 0:
                print(f"[{args.dataset}] progress: {completed} done, {failed} failed, {len(indices) - completed - failed} left")

    print(f"[{args.dataset}] finished: {completed} captioned, {failed} failed, output={output_path}")


if __name__ == "__main__":
    main()
