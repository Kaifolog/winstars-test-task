import argparse
from typing import Union

from PIL import Image, JpegImagePlugin

from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
import torch


def imgcls_inference(model: str, input_path: str, local_model: bool) -> str:

    image = Image.open(input_path)
    if not isinstance(image, JpegImagePlugin.JpegImageFile):
        raise TypeError("Not a JPEG file")

    image_processor = AutoImageProcessor.from_pretrained(
        model, local_files_only=local_model
    )
    inputs = image_processor(image, return_tensors="pt")

    model = AutoModelForImageClassification.from_pretrained(
        model, local_files_only=local_model
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label].lower()


def main():
    parser = argparse.ArgumentParser(
        description="Performs image classification to identify animal species."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="andriibul/animal-imgcls",
        help="Model name (default: 'andriibul/animal-imgcls')",
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument(
        "--local_model",
        action="store_true",
        help="Use local model instead of downloading from hub",
    )
    args = parser.parse_args()

    result = imgcls_inference(args.model, args.input, args.local_model)
    print(result)


if __name__ == "__main__":
    main()
