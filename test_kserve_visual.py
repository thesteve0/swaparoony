#!/usr/bin/env python3
"""
Test script for KServe face swap API that saves response images as viewable files.
"""

import requests
import base64
import json
from pathlib import Path
from PIL import Image
from io import BytesIO

# Configuration
INPUT_IMAGE_PATH = "headshot.webp"  # Change this to your test image
KSERVE_URL = "http://localhost:8080/v1/models/swaparoony-face-swap:predict"
OUTPUT_DIR = Path(".")  # Current directory


def load_and_encode_image(image_path: str) -> str:
    """Load image and convert to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def decode_and_save_image(image_b64: str, filename: str):
    """Decode base64 image and save as WebP"""
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_b64)

        # Open with PIL and save as WebP
        image = Image.open(BytesIO(image_bytes))
        output_path = OUTPUT_DIR / f"{filename}.webp"
        image.save(output_path, "WEBP", quality=90)
        print(f"✓ Saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Failed to save {filename}: {e}")
        return None


def main():
    # Check if input image exists
    if not Path(INPUT_IMAGE_PATH).exists():
        print(f"Error: Input image not found: {INPUT_IMAGE_PATH}")
        print(
            "Please update INPUT_IMAGE_PATH in the script to point to your test image"
        )
        return

    print(f"Loading image: {INPUT_IMAGE_PATH}")

    try:
        # Load and encode input image
        image_b64 = load_and_encode_image(INPUT_IMAGE_PATH)
        print(f"Image encoded ({len(image_b64)} chars)")

        # Prepare request
        request_data = {
            "image": image_b64,
            "source_face_id": 1,
            "destination_face_id": 1,
        }

        print(f"Sending request to KServe...")

        # Send request
        response = requests.post(KSERVE_URL, json=request_data, timeout=180)

        print(f"Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return

        # Parse response
        result = response.json()

        if not result.get("success", False):
            print(f"Face swap failed: {result.get('error', 'Unknown error')}")
            print(f"Detail: {result.get('detail', 'No details')}")
            return

        # Extract and save images
        swapped_images = result.get("swapped_images", [])
        faces_detected = result.get("faces_detected_in_source", 0)

        print(f"\n✓ Face swap successful!")
        print(f"✓ Faces detected in source: {faces_detected}")
        print(f"✓ Generated {len(swapped_images)} swapped images")

        print(f"\nSaving images to {OUTPUT_DIR.absolute()}:")

        saved_count = 0
        for i, img_data in enumerate(swapped_images):
            image_b64 = img_data.get("image_data", "")
            dest_name = img_data.get("destination_name", f"unknown_{i}")

            # Create filename (remove extension, add number)
            base_name = Path(dest_name).stem
            filename = f"swapped_{i+1:02d}_{base_name}"

            if decode_and_save_image(image_b64, filename):
                saved_count += 1

        print(f"\n🎉 Successfully saved {saved_count}/{len(swapped_images)} images!")
        print(f"You can now view the WebP files in: {OUTPUT_DIR.absolute()}")

    except requests.exceptions.Timeout:
        print(
            "Error: Request timed out (3 minutes). The model might be processing a large image or having issues."
        )
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to KServe. Is it running on localhost:8080?")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
