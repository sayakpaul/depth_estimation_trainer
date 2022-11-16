import argparse
import os
from datetime import datetime
from pprint import pformat

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import wandb
from imgaug import augmenters as iaa
from transformers import (
    GLPNFeatureExtractor,
    GLPNForDepthEstimation,
    Trainer,
    TrainingArguments,
)

from dataset import DIODEDataset

_TRAIN_DIR = "train_subset"
_VAL_DIR = "val"
_RESIZE_TO = (512, 512)
_TIMESTAMP = datetime.utcnow().strftime("%y%m%d-%H%M%S")


def compute_metrics(eval_pred):
    """Computes RMSE on a batch of predictions"""
    logits, labels = eval_pred
    rmse = (labels - logits) ** 2
    rmse = np.sqrt(rmse.mean())
    return {"rmse": rmse}


def collate_fn(examples):
    pixel_values = torch.stack([example["image"] for example in examples])
    labels = torch.stack([example["depth_map"].squeeze() for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a depth estimation model."
    )
    # Model related
    parser.add_argument(
        "--model_ckpt",
        default="vinvino02/glpn-nyu",
        type=str,
        choices=["vinvino02/glpn-nyu", "vinvino02/glpn-kitti"],
        help="Name of the GLPN model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--decoder_init", action="store_true", help="Initialize deocder randomly."
    )
    parser.add_argument(
        "--head_init", action="store_true", help="Initialize estimation head randomly."
    )
    # Training related
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    # Misc
    parser.add_argument("--run_name", default=None, type=str, help="WandB run name")
    parser.add_argument("--log_code", action="store_true")
    return parser.parse_args()


def prepare_data_df(dataset_root_path, variant="indoors"):
    path = os.path.join(dataset_root_path, variant)

    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith("_depth.npy")],
        "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
    }
    df = pd.DataFrame(data)
    return df


# Copied from transformers.models.segformer.modeling_segformer.SegformerPreTrainedModel._init_weights
def init_weights(model):
    def fn(module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    return fn


def main(args):
    print(f"Initializing model and feature extractor with {args.model_ckpt}...")
    feature_extractor = GLPNFeatureExtractor.from_pretrained(args.model_ckpt)
    model = GLPNForDepthEstimation.from_pretrained(args.model_ckpt)
    init_fn = init_weights(model)

    if args.head_init:
        print("Randomly initializing the head...")
        _ = model.head.apply(init_fn)
    if args.decoder_init:
        _ = model.decoder.apply(init_fn)

    print("Initializing WandB...")
    if args.run_name is None:
        run_name = f"de-ckpt@{args.model_ckpt}-head_init@{args.head_init}-decoder_init@{args.decoder_init}"
        run_name += f"-lr@{args.lr}-wd@{args.weight_decay}-wr@{args.warmup_ratio}"
        run_name += f"-{_TIMESTAMP}"
    else:
        run_name = args.run_name
    wandb.init(
        project="depth_estimation",
        name=run_name,
        entity="sayakpaul",
        config=vars(args),
    )
    if args.log_code:
        print("Logging code...")
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    print("Preparing dataframes...")
    train_df, val_df = prepare_data_df(_TRAIN_DIR), prepare_data_df(_VAL_DIR)

    print("Preparing augmentation chains...")
    # Heatmap transformations: https://imgaug.readthedocs.io/en/latest/source/examples_heatmaps.html
    train_transform_chain = iaa.Sequential(
        [
            iaa.Resize(_RESIZE_TO, interpolation="linear"),
            iaa.Fliplr(0.3),  # affects heatmaps
            iaa.Sharpen((0.0, 1.0), name="sharpen"),  # sharpen (only) image
            iaa.Sometimes(
                0.5, iaa.Affine(rotate=(-45, 45))
            ),  # rotate by -45 to 45 degrees (affects heatmaps)
            iaa.Sometimes(
                0.5, iaa.ElasticTransformation(alpha=50, sigma=5)
            ),  # apply water effect (affects heatmaps)
        ],
        random_order=True,
    )
    test_transformation_chain = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(_RESIZE_TO),
            torchvision.transforms.ToTensor(),
        ]
    )
    wandb.log({"train_augs": pformat(str(train_transform_chain))})

    print("Preparing datasets...")
    train_dataset = DIODEDataset(train_df, train_transform_chain, ["sharpen"])
    validation_dataset = DIODEDataset(val_df, test_transformation_chain)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    print("Initializing training args...")
    model_name = args.model_ckpt.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-finetuned-diode-{_TIMESTAMP}",  # To ensure there's a correspondence between WandB and Hub
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=25,
        load_best_model_at_end=True,
        push_to_hub=True,
        fp16=True,
    )
    wandb.config.update(training_args, allow_val_change=True)

    print("Initializing Trainer and training...")
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    _ = trainer.train()
    kwargs = {
        "tags": ["vision", "depth-estimation"],
        "finetuned_from": args.model_ckpt,
        "dataset": "diode-subset",
    }
    commit_message = f"wandb run name: {wandb.run.name}"
    trainer.push_to_hub(commit_message=commit_message, **kwargs)
    print("Training done and model pushed...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
