from __future__ import annotations

import argparse
import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
import tensorflow as tf

EXPECTED_CLASSES = 13_065
EXPECTED_IMAGES = 684_677
MIN_AUGMENTED_TRAIN_SAMPLES = 200
STANDARD_TRAIN_SAMPLES = 40
IMAGE_SIZE = 64


@dataclass(frozen=True)
class Sample:
    archive_name: str
    sample_number: int


@dataclass(frozen=True)
class ClassSamples:
    class_id: str
    label: int
    train: tuple[Sample, ...]
    test: tuple[Sample, ...]

    @property
    def repeat_count(self) -> int:
        return math.ceil(MIN_AUGMENTED_TRAIN_SAMPLES / len(self.train))

    @property
    def augmented_train_count(self) -> int:
        return len(self.train) * self.repeat_count


class HandwritingDataset:
    def __init__(self, archive_path: str | Path = "data/all_data.zip") -> None:
        self.archive_path = Path(archive_path)
        self.classes = self._build_index()
        self.num_classes = len(self.classes)
        self.original_train_size = sum(len(c.train) for c in self.classes)
        self.test_size = sum(len(c.test) for c in self.classes)
        self.augmented_train_size = sum(c.augmented_train_count for c in self.classes)

    def _build_index(self) -> tuple[ClassSamples, ...]:
        grouped: dict[str, list[tuple[int, str]]] = {}
        with ZipFile(self.archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                parts = PurePosixPath(info.filename).parts
                if len(parts) < 3 or parts[0] != "cleaned_data" or not parts[1].isdigit():
                    raise ValueError(f"Unexpected dataset path: {info.filename!r}")
                match = re.search(r"_(\d+)\.png$", parts[-1], re.IGNORECASE)
                if match is None:
                    raise ValueError(f"Unexpected image filename: {info.filename!r}")
                grouped.setdefault(parts[1], []).append((int(match.group(1)), info.filename))

        if len(grouped) != EXPECTED_CLASSES or sum(map(len, grouped.values())) != EXPECTED_IMAGES:
            raise ValueError("The dataset does not contain the expected 13,065 classes and 684,677 images")

        indexed_classes = []
        for label, class_id in enumerate(sorted(grouped, key=int)):
            ordered = sorted(grouped[class_id], key=lambda item: (item[0], item[1]))
            train_count = (
                STANDARD_TRAIN_SAMPLES
                if len(ordered) >= STANDARD_TRAIN_SAMPLES
                else max(1, round(len(ordered) * 0.8))
            )
            split = [Sample(name, number) for number, name in ordered]
            indexed_classes.append(
                ClassSamples(
                    class_id=class_id,
                    label=label,
                    train=tuple(split[:train_count]),
                    test=tuple(split[train_count:]),
                )
            )
        return tuple(indexed_classes)

    def validate(self) -> None:
        if self.num_classes != EXPECTED_CLASSES:
            raise ValueError(f"Expected {EXPECTED_CLASSES} classes, found {self.num_classes}")
        if self.original_train_size + self.test_size != EXPECTED_IMAGES:
            raise ValueError("The train/test split does not include every image exactly once")
        if any(not c.test for c in self.classes):
            raise ValueError("Every class must contain at least one testing image")
        if any(c.augmented_train_count < MIN_AUGMENTED_TRAIN_SAMPLES for c in self.classes):
            raise ValueError("Every class must contain at least 200 augmented training samples")

    def summary(self) -> dict[str, int]:
        return {
            "classes": self.num_classes,
            "images": self.original_train_size + self.test_size,
            "original_train_images": self.original_train_size,
            "test_images": self.test_size,
            "augmented_train_images": self.augmented_train_size,
            "minimum_augmented_train_images_per_class": min(
                c.augmented_train_count for c in self.classes
            ),
            "classes_with_fewer_than_40_images": sum(
                len(c.train) + len(c.test) < STANDARD_TRAIN_SAMPLES for c in self.classes
            ),
        }

    @staticmethod
    def read_image(archive: ZipFile, sample: Sample) -> np.ndarray:
        data = np.frombuffer(archive.read(sample.archive_name), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot decode image: {sample.archive_name!r}")
        return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    @staticmethod
    def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        height, width = image.shape
        center = ((width - 1) / 2, (height - 1) / 2)
        matrix = cv2.getRotationMatrix2D(
            center,
            angle=float(rng.uniform(-10, 10)),
            scale=float(rng.uniform(0.9, 1.1)),
        )
        matrix[0, 1] += float(rng.uniform(-0.12, 0.12))
        matrix[0, 2] += float(rng.uniform(-0.03, 0.03) * width)
        matrix[1, 2] += float(rng.uniform(-0.03, 0.03) * height)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        return (image.astype(np.float32) / 255.0)[..., np.newaxis]

    def _training_iterator(self, seed: int | None, counter: itertools.count):
        rng = np.random.default_rng(None if seed is None else seed + next(counter))
        with ZipFile(self.archive_path) as archive:
            for class_samples in self.classes:
                for _ in range(class_samples.repeat_count):
                    for sample in class_samples.train:
                        image = self.read_image(archive, sample)
                        yield self._normalize(self.augment(image, rng)), class_samples.label

    def _testing_iterator(self):
        with ZipFile(self.archive_path) as archive:
            for class_samples in self.classes:
                for sample in class_samples.test:
                    image = self.read_image(archive, sample)
                    yield self._normalize(image), class_samples.label

    def as_tf_dataset(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle_buffer: int = 4_096,
        seed: int | None = 42,
    ) -> tf.data.Dataset:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        output_signature = (
            tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
        if split == "train":
            counter = itertools.count()
            dataset = tf.data.Dataset.from_generator(
                lambda: self._training_iterator(seed, counter),
                output_signature=output_signature,
            )
            dataset = dataset.shuffle(
                shuffle_buffer, seed=seed, reshuffle_each_iteration=True
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                self._testing_iterator, output_signature=output_signature
            )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def save_augmentation_preview(self, output_path: str | Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)
        selected = [self.classes[i] for i in np.linspace(0, self.num_classes - 1, 6, dtype=int)]
        figure, axes = plt.subplots(len(selected), 6, figsize=(12, 12))
        with ZipFile(self.archive_path) as archive:
            for row, class_samples in enumerate(selected):
                original = self.read_image(archive, class_samples.train[0])
                for column in range(6):
                    image = original if column == 0 else self.augment(original, rng)
                    axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=255)
                    axes[row, column].axis("off")
                    if row == 0:
                        axes[row, column].set_title("Original" if column == 0 else f"Augmented {column}")
                    if column == 0:
                        axes[row, column].set_ylabel(f"Class {class_samples.class_id}")
        figure.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and augment the handwriting dataset")
    parser.add_argument("--archive", default="data/all_data.zip")
    parser.add_argument("--preview", default="outputs/augmentation_preview.png")
    args = parser.parse_args()

    dataset = HandwritingDataset(args.archive)
    dataset.validate()
    for key, value in dataset.summary().items():
        print(f"{key}: {value:,}")
    dataset.save_augmentation_preview(args.preview)
    print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
