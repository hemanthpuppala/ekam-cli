"""Dataset loading and preparation utilities."""

import json
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, load_dataset
from loguru import logger


def load_dataset_from_file(
    file_path: Path,
    text_column: str = "text",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load dataset from JSON or CSV file.

    Args:
        file_path: Path to dataset file
        text_column: Column name containing text data
        max_samples: Limit samples for debugging

    Returns:
        HuggingFace Dataset

    Raises:
        ValueError: If file format not supported or missing columns
    """
    try:
        file_path = Path(file_path)

        if file_path.suffix == ".json":
            logger.info(f"Loading JSON dataset: {file_path}")
            with open(file_path) as f:
                data = json.load(f)

            if isinstance(data, list):
                dataset = Dataset.from_dict({text_column: [item.get(text_column, "") for item in data]})
            else:
                dataset = Dataset.from_dict(data)

        elif file_path.suffix == ".csv":
            logger.info(f"Loading CSV dataset: {file_path}")
            dataset = load_dataset("csv", data_files=str(file_path))
            dataset = dataset["train"]

        elif file_path.suffix == ".jsonl":
            logger.info(f"Loading JSONL dataset: {file_path}")
            data = [json.loads(line) for line in open(file_path)]
            dataset = Dataset.from_dict({text_column: [item.get(text_column, "") for item in data]})

        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Limited to {max_samples} samples")

        logger.info(f"✓ Loaded {len(dataset)} samples from {file_path.name}")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        raise


def load_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load
        text_column: Column name containing text
        max_samples: Limit samples

    Returns:
        HuggingFace Dataset

    Raises:
        Exception: If dataset loading fails
    """
    try:
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)

        if text_column not in dataset.column_names:
            raise ValueError(f"Column '{text_column}' not found in dataset")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Limited to {max_samples} samples")

        logger.info(f"✓ Loaded {len(dataset)} samples from {dataset_name}")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset: {e}", exc_info=True)
        raise


def tokenize_function(examples: dict, tokenizer: Any, max_length: int = 512) -> dict:
    """Tokenize text examples.

    Args:
        examples: Batch of examples with text data
        tokenizer: HuggingFace tokenizer
        max_length: Max token length

    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def load_dataset(
    config: "DataConfig",
    tokenizer: Any,
    max_length: int = 512,
) -> tuple[Dataset, Dataset]:
    """Load and prepare dataset with train/eval split.

    Args:
        config: DataConfig with dataset settings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum token length

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    try:
        # Load dataset
        if config.dataset_path:
            dataset = load_dataset_from_file(
                config.dataset_path,
                text_column=config.text_column,
                max_samples=config.max_samples,
            )
        elif config.dataset_name:
            dataset = load_huggingface_dataset(
                config.dataset_name,
                split=config.split,
                text_column=config.text_column,
                max_samples=config.max_samples,
            )
        else:
            raise ValueError("Either dataset_path or dataset_name must be provided")

        # Tokenize dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer, max_length),
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=4,
        )

        # Split into train/eval
        if config.val_split > 0:
            split_dataset = tokenized_dataset.train_test_split(test_size=config.val_split)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = tokenized_dataset
            eval_dataset = tokenized_dataset

        logger.info(f"✓ Dataset prepared: {len(train_dataset)} train, {len(eval_dataset)} eval")
        return train_dataset, eval_dataset

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}", exc_info=True)
        raise
