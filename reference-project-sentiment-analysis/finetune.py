import argparse
import itertools
from functools import partial

import evaluate
import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from datasets.formatting.formatting import LazyBatch
from transformers import (
    BatchEncoding,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
import mlflow.transformers

TEXT_COL = "text"
LABEL_COL = "label"
TRAIN = "train"
TEST = "test"
VAL = "validation"
DEFAULT_DATASET = "ben-epstein/amazon_polarity_10_pct"

EVAL_METRIC = "f1"
METRIC = evaluate.load(EVAL_METRIC)

MODEL_DIR = "/mnt/artifacts/amazon-sentiment/"


def load_data(
    name: str = DEFAULT_DATASET, text_col: str = "content", label_col: str = "label"
) -> DatasetDict:
    """Load dataset from hub, map to standard column names, drop extra columns and keys that don't conform"""
    ds = load_dataset(name)
    if TRAIN not in ds:
        raise ValueError(
            f"Dataset must have a {TRAIN} split, optionally including {TEST} and {VAL}"
        )
    ds = ds.rename_columns({text_col: TEXT_COL, label_col: LABEL_COL})
    all_columns = set(itertools.chain(*ds.column_names.values()))
    unneeded_cols = [c for c in all_columns if c not in {TEXT_COL, LABEL_COL}]
    ds = ds.remove_columns(unneeded_cols)
    ds = DatasetDict({k: v for k, v in ds.items() if k in [TRAIN, TEST, VAL]})
    if not isinstance(ds[TRAIN].features[LABEL_COL], ClassLabel):
        print(f"Label {LABEL_COL} is not a ClassLabel. Encoding labels for training")
        ds = ds.class_encode_column(LABEL_COL)
        print(f"Encoded {LABEL_COL} as {ds[TRAIN].features[LABEL_COL]}")
    return ds


def split(ds: DatasetDict) -> DatasetDict:
    """Split the dataset into train, test, and val. If already present, return

    Guarantee that `train` will be one of the splits, no other guarantees
    """
    if all([TRAIN in ds, TEST in ds, VAL in ds]):
        print("All splits already available, returning ds")
        return ds
    elif TRAIN in ds and TEST in ds:
        # We split train into train/val, and keep test separate
        ds_train_val = ds[TRAIN].train_test_split(
            seed=42, test_size=0.1, stratify_by_column=LABEL_COL
        )
        data = DatasetDict(
            {TRAIN: ds_train_val[TRAIN], VAL: ds_train_val[TEST], TEST: ds[TEST]}
        )
    elif TRAIN in ds and VAL in ds:
        # We split train into train/test, and keep val (as it was specified by the user)
        ds_train_test = ds[TRAIN].train_test_split(
            seed=42, test_size=0.1, stratify_by_column=LABEL_COL
        )
        data = DatasetDict(
            {TRAIN: ds_train_test[TRAIN], VAL: ds[VAL], TEST: ds_train_test[TEST]}
        )
    else:
        # We only have a train. Split twice for test and val
        ds_train_test = ds[TRAIN].train_test_split(
            seed=42, test_size=0.1, stratify_by_column=LABEL_COL
        )
        # Further split the new 'train' split into train/val
        ds_train_val = ds_train_test[TRAIN].train_test_split(
            seed=42, test_size=0.1, stratify_by_column=LABEL_COL
        )
        data = DatasetDict(
            {
                TRAIN: ds_train_val[TRAIN],
                VAL: ds_train_val[VAL],
                TEST: ds_train_test[TEST],
            }
        )

    print(f"Samples in train      : {len(data[TRAIN])}")
    print(f"Samples in validation : {len(data[VAL])}")
    print(f"Samples in test       : {len(data[TEST])}")

    return data


def preprocess_function(
    tokenizer: DistilBertTokenizer, examples: LazyBatch
) -> BatchEncoding:
    return tokenizer(
        examples[TEXT_COL], truncation=True, padding=False, max_length=512
    )  # 512 because we use BERT


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    predictions, labels = np.array(eval_pred.predictions), np.array(eval_pred.label_ids)
    predictions = predictions.argmax(axis=1)
    return METRIC.compute(
        predictions=predictions, references=labels, average="weighted"
    )


def train(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    args: TrainingArguments,
    dataset_train: Dataset,
    dataset_val: Dataset,
) -> Trainer:

    mlflow.transformers.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=False,
        log_datasets=False
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.1
            )
        ],
    )

    trainer.train()

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tuning a distil-BERT model using Sentiment Analysis from Amazon product reviews. "
            "This work is licensed under the Creative Commons Attribution 4.0 International License."
        )
    )

    parser.add_argument(
        "--lr", help="Learning rate.", required=False, default=0.00001, type=float
    )
    parser.add_argument(
        "--epochs", help="Training epochs.", required=False, default=1, type=int
    )
    parser.add_argument(
        "--train_batch_size",
        help="Training batch size.",
        required=False,
        default=32,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        help="Eval batch size.",
        required=False,
        default=32,
        type=int,
    )
    parser.add_argument(
        "--dataset_name",
        help="The dataset name on huggingface hub",
        required=False,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--text_col",
        help="Column in the dataset of the text data",
        required=False,
        default="content",
    )
    parser.add_argument(
        "--label_col",
        help="Column in the dataset of the text data",
        required=False,
        default="label",
    )
    parser.add_argument(
        "--distilbert_model",
        help="Distil-BERT model from huggingface hub to use",
        required=False,
        default="distilbert-base-uncased",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("GPU acceleration is available!")
    else:
        print(
            "GPU acceleration is NOT available! Training, fine-tuning, and inference speed will be adversely impacted."
        )

    ds = load_data(args.dataset_name, args.text_col, args.label_col)
    ds = split(ds)

    model = DistilBertForSequenceClassification.from_pretrained(
        args.distilbert_model, id2label={0: 'negative', 1: 'positive'}
    )
    tokenizer = DistilBertTokenizer.from_pretrained(args.distilbert_model)

    ds = ds.map(partial(preprocess_function, tokenizer))

    train_args = TrainingArguments(
        output_dir="temp/",
        evaluation_strategy="steps",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        metric_for_best_model="f1",
        save_total_limit=2,
        save_strategy="steps",
        load_best_model_at_end=True,
        optim="adamw_torch",
    )

    trainer = train(model, tokenizer, train_args, ds[TRAIN], ds[VAL])

    eval_test = trainer.predict(ds[TEST])
    print(f"Performance on test: {eval_test.metrics}")

    # Please change the location to where you want to save the model, /mnt/artifacts is available for git based projects
    trainer.save_model(MODEL_DIR)


if __name__ == "__main__":
    main()
