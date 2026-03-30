"""
Self-Recaptioning Pipeline for SafeGuard-VL (Stage-1 Data Construction)

Two-step process:
  1. Baseline model (Qwen2.5-VL) produces the original caption.
  2. Recaptioning model (e.g., Gemma) recovers suppressed unsafe semantics
     via minimal edits, producing a caption with more unsafe details.

The recaptioned data is used for Stage-1 SFT training.

Usage:
    python recaption.py \
        --input_jsonl /path/to/captions.jsonl \
        --output_jsonl /path/to/recaptioned.jsonl \
        --api_key YOUR_API_KEY \
        --base_url YOUR_BASE_URL \
        --model gemma-27b
"""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

RECAPTION_PROMPT = """You are a professional content security reviewer, skilled in accurately identifying and detailing various violations and harmful content in images.

For input images, please process them according to the following steps:

1. **Internal Analysis (No Output)**:
   - First, comprehensively determine the appropriate review tag for the image (e.g., violence, pornography, terrorism, child safety, illegal activities, hate symbols, self-harm, drug abuse, etc.).
   - Then, systematically list all specific harmful points (a), including but not limited to:
     - The violating subject (person, object, symbol, text, etc.) and its attributes (age, gender, clothing, posture, injuries, degree of nudity, etc.);
     - The scene environment (indoor/outdoor, location features, background elements);
     - Behavior (whether it involves aggression, sexual innuendo, self-harm, illegal transactions, etc.);
     - Visual elements (blood, nudity, weapons, drugs, extremist symbols, etc.);
     - Potentially suggestive or beautifying expressions of violating behavior;
     - Any content that may violate laws and regulations, platform policies, or social norms.

2. **Output Integration (Output this part only)**:
   - Based on the given caption (i.e., "{caption}"), deeply integrate it with the harmful points (a) mentioned above;
   - **Retain and expand** the harmful point-related information in the caption, supplementing specific details (such as location, degree, context, intent, etc.);
   - **Delete or ignore** descriptions in the caption that are completely unrelated to the review tags and harmful points (such as unrelated scenery, neutral objects, unrelated people, etc.);
   - Finally, output a coherent, objective, and highly detailed description, focusing on all violations and harmful content in the image, **without adding any explanations, conclusions, suggestions, or additional formatting**, outputting only a plain text description.

Please strictly follow the above instructions and only output the integrated detailed harmful content description."""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def recaption_single(item: dict, api_key: str, base_url: str, model: str) -> dict:
    """Recaption a single item using the LLM API."""
    image_path = item.get("image", "")
    caption = item.get("answer", "")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    encoded = encode_image(image_path)
    prompt = RECAPTION_PROMPT.format(caption=caption)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    resp = requests.post(
        f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60
    )
    resp.raise_for_status()
    result = resp.json()
    new_caption = result["choices"][0]["message"]["content"].strip()

    output = item.copy()
    output["answer"] = new_caption
    return output


def main():
    parser = argparse.ArgumentParser(description="Self-recaptioning pipeline")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--model", type=str, default="gemma-27b")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    # Load data
    data = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} items")

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    success, failed = 0, 0
    with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for item in data:
                fut = executor.submit(
                    recaption_single, item, args.api_key, args.base_url, args.model
                )
                futures[fut] = item

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Recaptioning"):
                try:
                    result = fut.result()
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    success += 1
                except Exception as e:
                    print(f"Error: {e}")
                    failed += 1

    print(f"Done! Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
