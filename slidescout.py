import os
import argparse
import base64
import re
import sys
import cv2
from openai import OpenAI
from PIL import Image
from contextlib import redirect_stdout

# Initialize OpenAI client with API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("The OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=api_key)

def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def get_slide_prompt(slide_num, total_slides, audience, story_so_far=None):
    intro = (
        f"You are analyzing slide {slide_num} of {total_slides} from a scientific or medical presentation.\n"
        f"Your task is to extract every bit of information from each slide in text form.\n"
        f"I want you to capture everything and store it in a structured way.\n"
        f"I want you to extract everything from figures, tables, images, text, everything.\n"
        f"The goal is to have a comprehensive database of everything in this presentation by the final slide.\n"
    )
    if story_so_far:
        intro += (
            f"\nCurrent story so far:\n{story_so_far}\n"
            f"\nNow add to this story by analyzing the new content from this slide."
        )
    else:
        intro += "\nStart the narrative based on this first slide."
    return intro

def natural_key(string):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string)]

def crop_slide_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_contour = None
            max_area = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0

                if area > 50000 and 1.2 < aspect_ratio < 2.0 and area > max_area:
                    best_contour = (x, y, w, h)
                    max_area = area

            if best_contour:
                x, y, w, h = best_contour
                padding = 10
                x = max(x - padding, 0)
                y = max(y - padding, 0)
                w = min(w + 2 * padding, image.shape[1] - x)
                h = min(h + 2 * padding, image.shape[0] - y)

                cropped = image[y:y+h, x:x+w]

                if cropped.shape[0] > cropped.shape[1]:
                    cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

                out_path = os.path.join(output_folder, filename)
                cv2.imwrite(out_path, cropped)
                print(f"Cropped and saved: {out_path}")
            else:
                print(f"⚠️ Skipped {filename}: No suitable contour found.")

def summarize_presentation_from_images(folder_path, audience):
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=natural_key
    )
    if not image_files:
        raise ValueError("No image files found in the specified folder.")

    total_slides = len(image_files)
    story_so_far = None

    for i, filename in enumerate(image_files):
        slide_path = os.path.join(folder_path, filename)
        print(f"\n=== Processing slide {i+1}/{total_slides}: {filename} ===")

        b64_image = encode_image_base64(slide_path)
        prompt = get_slide_prompt(i+1, total_slides, audience, story_so_far)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            story_so_far = response.choices[0].message.content.strip()
            print(f"\n--- Narrative so far (after slide {i+1}) ---\n{story_so_far}\n")
        except Exception as e:
            raise RuntimeError(f"Error processing slide {i+1} ({filename}): {e}")

    return story_so_far

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def write_summary_to_file(summary, output_file):
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(summary)
    except Exception as e:
        raise RuntimeError(f"Error writing summary to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop slide images and summarize them using GPT-4o.")
    parser.add_argument("--input", required=True, help="Path to the folder containing raw slide images.")
    parser.add_argument("--cropped", required=True, help="Path to the folder for cropped slide outputs.")
    parser.add_argument("--audience", default="HCP", choices=["patient", "HCP", "stakeholder", "training"], help="Target audience for the summary.")
    parser.add_argument("--output", required=False, help="Optional output file to save the summary.")
    args = parser.parse_args()

    crop_slide_images(args.input, args.cropped)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            with redirect_stdout(Tee(sys.stdout, f)):
                try:
                    summary = summarize_presentation_from_images(args.cropped, args.audience)
                    print("\nFinal Presentation Narrative:\n")
                    print(summary)
                except Exception as e:
                    print(f"Error: {e}")
    else:
        try:
            summary = summarize_presentation_from_images(args.cropped, args.audience)
            print("\nFinal Presentation Narrative:\n")
            print(summary)
        except Exception as e:
            print(f"Error: {e}")
