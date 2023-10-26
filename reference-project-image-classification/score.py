import argparse
from typing import Dict, List
from PIL import Image

from transformers import pipeline
import pandas as pd


MODEL = "Domino-ai/vit-base-patch16-224-in21k-food101"
PIPELINE = pipeline("image-classification", model=MODEL)
DEFAULT_IMAGE_PATH = "burger.png"


def predict_image(image: Image) -> List[Dict]:
    results = PIPELINE(image)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate your fine-tuned model")
    parser.add_argument(
        "--eval_path",
        help="Path to eval data image file. Assumed to be a png or jpeg",
        required=False,
    )
    args = parser.parse_args()
    image_path = args.eval_path if args.eval_path else DEFAULT_IMAGE_PATH
    image = Image.open(image_path)
    classes = predict_image(image)
    print("*" * 50)
    print("Predictions")
    print(pd.DataFrame(classes)[["label", "score"]])


if __name__ == "__main__":
    main()
