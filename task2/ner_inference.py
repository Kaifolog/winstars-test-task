import argparse
from typing import Union
import torch
from transformers import DebertaV2TokenizerFast, AutoModelForTokenClassification


def ner_inference(
    model: str, input_string: str, multiple: bool, local_model: bool, verbose: bool
) -> Union[str, list[str], None]:
    tokenizer = DebertaV2TokenizerFast.from_pretrained(
        model, local_files_only=local_model
    )
    inputs = tokenizer(input_string, return_tensors="pt")

    model = AutoModelForTokenClassification.from_pretrained(
        model, local_files_only=local_model
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = torch.argmax(logits, dim=2)

    if verbose:
        print([tokenizer.decode(t) for t in inputs["input_ids"][0][1:-1]])
        print([model.config.id2label[t.item()] for t in predictions[0][1:-1]])

    predictions_list = predictions.tolist()[0][1:-1]
    deduplicated = list(dict.fromkeys(predictions_list))
    labels = [model.config.id2label[label][2:].lower() for label in deduplicated]

    if "o" in labels:
        labels.remove("o")

    if not multiple:
        if labels != []:
            labels = labels[0]
        else:
            None

    return labels


def main():
    parser = argparse.ArgumentParser(
       description="Performs NER to identify animal species."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="andriibul/animals-ner",
        help="Model name (default: 'andriibul/animals-ner')",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Text to process"
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Extract multiple animal species instead of just one",
    )
    parser.add_argument(
        "--local_model",
        action="store_true",
        help="Use local model instead of downloading from hub",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed output")
    args = parser.parse_args()

    result = ner_inference(
        args.model, args.input, args.multiple, args.local_model, args.verbose
    )
    print(result)


if __name__ == "__main__":
    main()
