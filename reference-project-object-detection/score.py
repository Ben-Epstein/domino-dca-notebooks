import argparse
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes
from transformers import pipeline

MODEL = "devonho/detr-resnet-50_finetuned_cppe5"
PIPELINE = pipeline("object-detection", model=MODEL)
DEFAULT_IMAGE_PATH = "mask.jpeg"


def predict_obejcts(image: Image) -> List[Dict]:
    results = PIPELINE(image)
    return results


def draw_boxes(image: Image, boxes_df: pd.DataFrame) -> Image:
    """Draws boxes at the specified coordinates on the image"""
    boxes_xyxy = boxes_df[["xmin", "ymin", "xmax", "ymax"]].values
    labels = boxes_df["label"].tolist()
    box_image = to_pil_image(
        draw_bounding_boxes(
            pil_to_tensor(image),
            torch.Tensor(boxes_xyxy),
            labels=labels,
            width=2,
            font="Arial.tff",
            font_size=75,
        )
    )
    return box_image


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
    objects = predict_obejcts(image)
    objects_df = pd.DataFrame(objects)
    objects_df[["xmin", "ymin", "xmax", "ymax"]] = objects_df.pop("box").apply(
        pd.Series
    )
    print("*" * 50)
    print("Predictions")
    print(objects_df)
    box_image = draw_boxes(image, objects_df)
    try:
        box_image.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
import argparse
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes
from transformers import pipeline

MODEL = "devonho/detr-resnet-50_finetuned_cppe5"
PIPELINE = pipeline("object-detection", model=MODEL)
DEFAULT_IMAGE_PATH = "mask.jpeg"


def predict_obejcts(image: Image) -> List[Dict]:
    results = PIPELINE(image)
    return results


def draw_boxes(image: Image, boxes_df: pd.DataFrame) -> Image:
    """Draws boxes at the specified coordinates on the image"""
    boxes_xyxy = boxes_df[["xmin", "ymin", "xmax", "ymax"]].values
    labels = boxes_df["label"].tolist()
    box_image = to_pil_image(
        draw_bounding_boxes(
            pil_to_tensor(image),
            torch.Tensor(boxes_xyxy),
            labels=labels,
            width=2,
            font="Arial",
            font_size=75,
        )
    )
    return box_image


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
    objects = predict_obejcts(image)
    objects_df = pd.DataFrame(objects)
    objects_df[["xmin", "ymin", "xmax", "ymax"]] = objects_df.pop("box").apply(
        pd.Series
    )
    print("*" * 50)
    print("Predictions")
    print(objects_df)
    box_image = draw_boxes(image, objects_df)
    try:
        box_image.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
