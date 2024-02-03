import os
import glob
from typing import Optional

import torch
import pytorchvideo.data
import numpy as np
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
)

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
    HfArgumentParser,
    Trainer,
    set_seed,
    pipeline,
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
    test_dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory of testing dataset. This will override dataset_dir."
        },
    )
    test_video_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path of a videao for inferencing. This will override test_dataset_dir and dataset_dir."
        },
    )
    sample_rate: int = field(
        default=4,
        metadata={"help": "clip_duration = num_frames_to_sample * sample_rate / fps."},
    )
    fps: int = field(
        default=30,
        metadata={
            "help": "FPS of testing data, clip_duration = num_frames_to_sample * sample_rate / fps."
        },
    )

    def __post_init__(self):
        if (
            self.test_video_path is None
            and self.test_dataset_dir is None
            and self.dataset_dir is None
        ):
            raise ValueError(
                "You must specify a testing dataset directory eithor by train_dataset_dir or dataset_dir. Or specify a testing video path."
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
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

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

    if data_args.test_video_path is not None:
        pipe = pipeline(model=model_args.model_name_or_path)
        print(pipe(data_args.test_video_path))
        return
    elif data_args.test_dataset_dir is not None:
        test_path = data_args.test_dataset_dir
    else:
        test_path = os.path.join(data_args.dataset_dir, "test")

    class_labels = sorted(os.listdir(test_path))
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Unique classes: {list(label2id.keys())}.")

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

    test_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path=test_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    def count_mp4_files(directory):
        mp4_files = glob.glob(os.path.join(directory, "**/*.mp4"), recursive=True)
        num_mp4_files = len(mp4_files)

        return num_mp4_files

    test_iter = iter(test_dataset)
    test_len = count_mp4_files(test_path)
    predictions = []
    labels = []
    errors = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for sample in tqdm(test_iter, total=test_len):
        print(f"Inferencing {sample['video_name']}.")
        video, label = sample["video"], sample["label"]
        perumuted_video = video.permute(1, 0, 2, 3)

        inputs = {"pixel_values": perumuted_video.unsqueeze(0).to(device)}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = logits.argmax(-1).item()
            predictions.append(pred)
            labels.append(label)

            if pred != label:
                errors.append(sample["video_name"])

    print(f"Accuracy: {accuracy_score(labels, predictions)}")
    print(f"F1 : {f1_score(labels, predictions, average=None)}")
    print(f"Average Precision: {average_precision_score(labels, predictions)}")
    print(f"error_list: {errors}")


if __name__ == "__main__":
    main()
