import mlflow
from omegaconf import DictConfig, OmegaConf
from os import environ

from datasets import load_dataset, DatasetDict, concatenate_datasets, ClassLabel
from transformers import (
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np


def train():
    cfg: DictConfig = OmegaConf.load("./ner_train.yaml")

    print(OmegaConf.to_yaml(cfg))

    synthetic = load_dataset("json", data_files="./datasets/animals_synthetic.json")
    facts = load_dataset("json", data_files="./datasets/animal-fun-facts.json")
    coco = load_dataset("json", data_files="./datasets/coco_animals.json")
    coco_filtered = load_dataset(
        "json", data_files="./datasets/coco_animals_filtered.json"
    )

    # dataset creation
    if cfg["dataset"] == "synthetic+facts":
        dataset_concat = (
            concatenate_datasets([synthetic["train"], facts["train"]])
            .shuffle(42)
            .train_test_split(test_size=0.2)
        )
        train_validate_split = dataset_concat["train"].train_test_split(
            test_size=0.15
        )  # 0.8*0.15=0.12
        dataset = DatasetDict(
            {
                "train": train_validate_split["train"],
                "validate": train_validate_split["test"],
                "test": dataset_concat["test"],
            }
        )
    elif cfg["dataset"] == "facts":
        facts_split = facts["train"].train_test_split(test_size=0.2)
        train_validate_split = facts_split["train"].train_test_split(
            test_size=0.15
        )  # 0.8*0.15=0.12
        dataset = DatasetDict(
            {
                "train": train_validate_split["train"],
                "validate": train_validate_split["test"],
                "test": facts_split["test"],
            }
        )
    elif cfg["dataset"] == "synthetic":
        synthetic_split = synthetic["train"].train_test_split(test_size=0.2)
        train_validate_split = synthetic_split["train"].train_test_split(
            test_size=0.15
        )  # 0.8*0.15=0.12
        dataset = DatasetDict(
            {
                "train": train_validate_split["train"],
                "validate": train_validate_split["test"],
                "test": synthetic_split["test"],
            }
        )
    else:
        raise Exception("dataset is not specified")

    class_labels = ClassLabel(names_file="./datasets/animal_classes.txt")

    if cfg["model"] == "google-bert/bert-base-cased":
        from transformers import BertTokenizerFast

        tokenizer = BertTokenizerFast.from_pretrained(
            cfg["model"], cache_dir=cfg["cache_folder"]
        )
    elif cfg["model"] == "FacebookAI/roberta-base":
        from transformers import RobertaTokenizerFast

        tokenizer = RobertaTokenizerFast.from_pretrained(
            cfg["model"], add_prefix_space=True, cache_dir=cfg["cache_folder"]
        )
    elif cfg["model"] == "microsoft/deberta-v3-base":
        from transformers import DebertaV2TokenizerFast

        tokenizer = DebertaV2TokenizerFast.from_pretrained(
            cfg["model"], cache_dir=cfg["cache_folder"]
        )
    else:
        raise Exception("tokenizer is not specified")

    # relabeling dataset after tokenization
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    seqeval = evaluate.load("seqeval")
    label_list = class_labels.names

    # setting evaluation metrics
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_coco_dataset = coco.map(tokenize_and_align_labels, batched=True)
    tokenized_coco_filtered_dataset = coco_filtered.map(
        tokenize_and_align_labels, batched=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    id2label = {i: class_labels.int2str(i) for i in range(len(class_labels.names))}
    label2id = {class_labels.int2str(i): i for i in range(len(class_labels.names))}

    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model"],
        num_labels=len(class_labels.names),
        id2label=id2label,
        label2id=label2id,
        cache_dir=cfg["cache_folder"],
    )

    environ["MLFLOW_EXPERIMENT_NAME"] = (
        f"{cfg['model']}-{cfg['dataset']}-{cfg['train']['num_epochs']}-{cfg['train']['learning_rate']}"
    )
    environ["MLFLOW_FLATTEN_PARAMS "] = "True"

    training_args = TrainingArguments(
        output_dir="ner_models",
        learning_rate=cfg["train"]["learning_rate"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        per_device_eval_batch_size=cfg["train"]["batch_size"],
        num_train_epochs=cfg["train"]["num_epochs"],
        weight_decay=cfg["train"]["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg["train"]["save_total_limit"],
        load_best_model_at_end=cfg["train"]["load_best_model_at_end"],
        metric_for_best_model=cfg["train"]["metric_for_best_model"],
        greater_is_better=cfg["train"]["greater_is_better"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validate"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_results = trainer.predict(test_dataset=tokenized_dataset["test"])
    print(test_results.metrics)
    mlflow.log_metrics(test_results.metrics)

    coco_test_results = trainer.predict(
        test_dataset=tokenized_coco_dataset["train"], metric_key_prefix="test_coco"
    )
    print(coco_test_results.metrics)
    mlflow.log_metrics(coco_test_results.metrics)

    coco_filtered_test_results = trainer.predict(
        test_dataset=tokenized_coco_filtered_dataset["train"],
        metric_key_prefix="test_coco_filtered",
    )
    print(coco_filtered_test_results.metrics)

    mlflow.end_run()


if __name__ == "__main__":
    train()
