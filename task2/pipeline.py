import argparse

from ner_inference import ner_inference
from imgcls_inference import imgcls_inference


def pipeline(
    ner_input_string: str,
    imgcls_input_path: str,
    ner_model: str = "andriibul/animals-ner",
    ner_multiple: bool = False,
    ner_local_model: bool = False,
    ner_verbose: bool = False,
    imgcls_model: str = "andriibul/animal-imgcls",
    imgcls_local_model: bool = False,
):
    ner_result = ner_inference(
        ner_model, ner_input_string, ner_multiple, ner_local_model, ner_verbose
    )

    if ner_result is None:
        return False

    imgcls_result = imgcls_inference(
        imgcls_model, imgcls_input_path, imgcls_local_model
    )
    if isinstance(ner_result, list):
        result_dict = {}
        for result in ner_result:
            result_dict[result] = bool(result == imgcls_result)
        return result_dict
    else:
        return ner_result == imgcls_result


def main():
    parser = argparse.ArgumentParser(description="Pipeline to identify animal species.")
    
    parser.add_argument(
        "--ner_input", type=str, required=True, help="Text to process for NER"
    )
    parser.add_argument(
        "--imgcls_input", type=str, required=True, help="Path to image file for classification"
    )

    parser.add_argument(
        "--ner_model",
        type=str,
        default="andriibul/animals-ner",
        help="NER model name (default: 'andriibul/animals-ner')",
    )
    parser.add_argument(
        "--ner_multiple",
        action="store_true",
        help="Extract multiple animal species instead of just one",
    )
    parser.add_argument(
        "--ner_local_model",
        action="store_true",
        help="Use local NER model instead of downloading from hub",
    )
    parser.add_argument(
        "--ner_verbose", action="store_true", help="Enable detailed NER output"
    )

    parser.add_argument(
        "--imgcls_model",
        type=str,
        default="andriibul/animal-imgcls",
        help="Image classification model name (default: 'andriibul/animal-imgcls')",
    )
    parser.add_argument(
        "--imgcls_local_model",
        action="store_true",
        help="Use local image classification model instead of downloading from hub",
    )
    args = parser.parse_args()

    print(
        pipeline(
            args.ner_input,
            args.imgcls_input,
            args.ner_model,
            args.ner_multiple,
            args.ner_local_model,
            args.ner_verbose,
            args.imgcls_model,
            args.imgcls_local_model,
        )
    )


if __name__ == "__main__":
    main()
