import os
import random
import warnings
from typing import Optional

import datasets
import evaluate
import torch
import pytorchvideo.data
import numpy as np
from dataclasses import dataclass, field
from datasets import load_dataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
    create_video_transform,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModelForVideoClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory of a dataset."},
    )
    train_dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory of training dataset. This will override dataset_dir."
        },
    )
    val_dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory of validating dataset. This will override dataset_dir."
        },
    )
    # max_train_samples: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "For debugging purposes or quicker training, truncate the number of training examples to this "
    #             "value if set."
    #         )
    #     },
    # )
    # max_eval_samples: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
    #             "value if set."
    #         )
    #     },
    # )
    sample_rate: int = field(
        default=4,
        metadata={"help": "clip_duration = num_frames_to_sample * sample_rate / fps."},
    )
    fps: int = field(
        default=30,
        metadata={
            "help": "FPS of training data, clip_duration = num_frames_to_sample * sample_rate / fps."
        },
    )

    def __post_init__(self):
        if self.train_dataset_dir is None and self.dataset_dir is None:
            raise ValueError(
                "You must specify a training dataset directory eithor by train_dataset_dir or dataset_dir."
            )
        if self.val_dataset_dir is None and self.dataset_dir is None:
            raise ValueError(
                "You must specify a valdating dataset directory eithor by val_dataset_dir or dataset_dir."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="MCG-NJU/videomae-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    # model_type: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "If training from scratch, pass a model type from the list: TimeSformer, VideoMAE, ViViT"
    #     },
    # )
    # config_name: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Pretrained config name or path if not the same as model_name"
    #     },
    # )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        model_args.token = model_args.use_auth_token

    if data_args.train_dataset_dir is None:
        train_path = os.path.join(data_args.dataset_dir, "train")
    else:
        train_path = data_args.train_dataset_dir

    if data_args.val_dataset_dir is None:
        val_path = os.path.join(data_args.dataset_dir, "val")
    else:
        val_path = data_args.val_dataset_dir

    class_labels = sorted(os.listdir(train_path))
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Unique classes: {list(label2id.keys())}.")

    set_seed(training_args.seed)

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForVideoClassification.from_pretrained(
        model_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames
    clip_duration = num_frames_to_sample * data_args.sample_rate / data_args.fps

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        Resize(resize_to),  # RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    train_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path=train_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    training_args.max_steps = (
        train_dataset.num_videos // training_args.per_device_train_batch_size
    ) * training_args.num_train_epochs

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    val_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path=val_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(examples):
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "video-classification",
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
