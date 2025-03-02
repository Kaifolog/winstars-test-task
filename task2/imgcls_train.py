from omegaconf import DictConfig, OmegaConf

from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor
from transformers import DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

import evaluate
import numpy as np
from collections import Counter

from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

cfg: DictConfig = OmegaConf.load("./imgcls_train.yaml")

print(OmegaConf.to_yaml(cfg))

dataset = load_dataset("Rapidata/Animals-10", split="train")

# downsampling
label_counts = Counter(dataset["label"])
min_class_size = min(label_counts.values())


def downsample_dataset(dataset):
    examples_by_class = {}
    for i, example in enumerate(dataset):
        label = example["label"]
        if label not in examples_by_class:
            examples_by_class[label] = []
        examples_by_class[label].append(i)

    selected_indices = []
    import random

    for label, indices in examples_by_class.items():
        selected_indices.extend(random.sample(indices, min_class_size))

    balanced_dataset = dataset.select(selected_indices)
    return balanced_dataset


dataset = downsample_dataset(dataset)

# split
dataset_split = dataset.train_test_split(test_size=0.2)
train_validate_split = dataset_split["train"].train_test_split(
    test_size=0.15
)  # validate: 0.8*0.15=0.12
dataset = DatasetDict(
    {
        "train": train_validate_split["train"],
        "validate": train_validate_split["test"],
        "test": dataset_split["test"],
    }
)


labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# preprocessing
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])


def transforms(examples):
    examples["pixel_values"] = [
        _transforms(img.convert("RGB")) for img in examples["image"]
    ]
    del examples["image"]
    return examples


dataset = dataset.with_transform(transforms)

data_collator = DefaultDataCollator()

# metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # axis=1 для классификации

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)[
            "accuracy"
        ],
        "f1": f1.compute(
            predictions=predictions, references=labels, average="weighted"
        )["f1"],
        "precision": precision.compute(
            predictions=predictions, references=labels, average="weighted"
        )["precision"],
        "recall": recall.compute(
            predictions=predictions, references=labels, average="weighted"
        )["recall"],
    }


model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)


training_args = TrainingArguments(
    output_dir="imgcls_models",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=cfg["train"]["learning_rate"],
    per_device_train_batch_size=cfg["train"]["batch_size"],
    per_device_eval_batch_size=cfg["train"]["batch_size"],
    gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
    warmup_ratio=cfg["train"]["warmup_ratio"],
    num_train_epochs=cfg["train"]["num_epochs"],
    weight_decay=cfg["train"]["weight_decay"],
    save_total_limit=cfg["train"]["save_total_limit"],
    load_best_model_at_end=cfg["train"]["load_best_model_at_end"],
    metric_for_best_model=cfg["train"]["metric_for_best_model"],
    greater_is_better=cfg["train"]["greater_is_better"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validate"],
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

test_results = trainer.predict(test_dataset=dataset["test"])
print(test_results.metrics)
