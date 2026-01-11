# ==================== Imports ====================
import argparse
import datetime
import os
import random
import time
import warnings

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from trainer import Trainer
from utils.utils import *
from utils.builders import build_model, build_dataloaders, get_class_info

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")


# ==================== Helper: str2bool ====================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(
    description="A highly configurable training script for RAER Dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# --- Experiment and Environment ---
exp_group = parser.add_argument_group("Experiment & Environment", "Basic settings for the experiment")
exp_group.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
exp_group.add_argument("--eval-checkpoint", type=str, default="", help="Path to model checkpoint for eval mode.")
exp_group.add_argument("--exper-name", type=str, default="test")
exp_group.add_argument("--dataset", type=str, default="RAER")
exp_group.add_argument("--gpu", type=str, default="0", help='GPU id, or "mps" or "cpu".')
exp_group.add_argument("--workers", type=int, default=4)
exp_group.add_argument("--seed", type=int, default=42)

# --- Data & Path ---
path_group = parser.add_argument_group("Data & Path", "Paths to datasets and pretrained models")
path_group.add_argument("--root-dir", type=str, default="./")
path_group.add_argument("--train-annotation", type=str, default="RAER/annotation/train.txt")
path_group.add_argument("--val-annotation", type=str, default="", help="Optional; if empty, will use test-annotation.")
path_group.add_argument("--test-annotation", type=str, default="RAER/annotation/test.txt")
path_group.add_argument("--clip-path", type=str, default="ViT-B/32")
path_group.add_argument("--bounding-box-face", type=str, default="RAER/bounding_box/face.json")
path_group.add_argument("--bounding-box-body", type=str, default="RAER/bounding_box/body.json")

# --- Training Control ---
train_group = parser.add_argument_group("Training Control", "Parameters to control the training process")
train_group.add_argument("--epochs", type=int, default=20)
train_group.add_argument("--batch-size", type=int, default=8)
train_group.add_argument("--print-freq", type=int, default=10)

# --- Optimizer & Learning Rate ---
optim_group = parser.add_argument_group("Optimizer & LR", "Hyperparameters for the optimizer and scheduler")
optim_group.add_argument("--lr", type=float, default=1e-2)
optim_group.add_argument("--lr-image-encoder", type=float, default=1e-5)
optim_group.add_argument("--lr-prompt-learner", type=float, default=1e-3)
optim_group.add_argument("--weight-decay", type=float, default=1e-4)
optim_group.add_argument("--momentum", type=float, default=0.9)
optim_group.add_argument("--milestones", nargs="+", type=int, default=[10, 15])
optim_group.add_argument("--gamma", type=float, default=0.1)

# --- Prompt set selector (THIS is what you wanted) ---
parser.add_argument(
    "--prompt_set",
    type=str,
    default="baseline",
    choices=["baseline", "student", "descriptor_face", "descriptor_body", "descriptor_full", "ensemble"],
    help="Choose which prompt group to use (baseline/student/descriptor/ensemble).",
)

# --- Model & Input ---
model_group = parser.add_argument_group("Model & Input", "Parameters for model architecture and data handling")
model_group.add_argument(
    "--text-type",
    default="class_descriptor",
    choices=[
        "class_names",
        "class_names_with_context",
        "class_descriptor",
        "class_descriptor_only_face",
        "class_descriptor_only_body",
        "prompt_ensemble",
    ],
    help="Type of text prompts to use.",
)
model_group.add_argument("--temporal-layers", type=int, default=1)
model_group.add_argument("--contexts-number", type=int, default=8)
model_group.add_argument("--class-token-position", type=str, default="end")
model_group.add_argument("--class-specific-contexts", type=str2bool, default=True)
model_group.add_argument("--load_and_tune_prompt_learner", type=str2bool, default=True)
model_group.add_argument("--num-segments", type=int, default=16)
model_group.add_argument("--duration", type=int, default=1)
model_group.add_argument("--image-size", type=int, default=224)

# --- Ensemble control (SAFE default) ---
model_group.add_argument(
    "--ensemble-k",
    type=int,
    default=3,
    help="How many prompts per class to use from prompt_ensemble_5 (only used if text-type=prompt_ensemble).",
)
model_group.add_argument(
    "--ensemble-mode",
    type=str,
    default="first",
    choices=["first", "flatten"],
    help=(
        "first: use only the first prompt per class (SAFE, total 5 prompts). "
        "flatten: use K prompts/class and flatten to 5*K prompts (requires model aggregation support)."
    ),
)


# ==================== Helper Functions ====================
def setup_environment(args: argparse.Namespace) -> argparse.Namespace:
    device_name = "cpu"
    if args.gpu == "mps":
        if torch.backends.mps.is_available():
            device_name = "mps"
        else:
            print("MPS not available, falling back to CPU")
    elif torch.cuda.is_available() and str(args.gpu).isdigit():
        device_name = f"cuda:{args.gpu}"

    args.device = torch.device(device_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = True
    print("Environment and random seeds set successfully.")
    return args


def setup_paths_and_logging(args: argparse.Namespace) -> argparse.Namespace:
    now = datetime.datetime.now()
    time_str = now.strftime("-[%m-%d]-[%H%M]")
    args.name = args.exper_name + time_str
    args.output_path = os.path.join("outputs", args.name)
    os.makedirs(args.output_path, exist_ok=True)

    print("************************")
    print("Running with the following configuration:")
    for k, v in vars(args).items():
        print(f"{k} = {v}")
    print("************************")

    log_txt_path = os.path.join(args.output_path, "log.txt")
    with open(log_txt_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k} = {v}\n")
        f.write("*" * 50 + "\n\n")

    return args


def _resolve_input_text_for_ensemble(args, input_text):
    """
    get_class_info() will return prompt_ensemble_5 as list[list[str]] for text-type=prompt_ensemble.
    Here we convert it into SAFE input for current model when ensemble-mode=first.
    """
    if args.text_type != "prompt_ensemble":
        return input_text

    # Expect list[list[str]]
    if not (isinstance(input_text, list) and len(input_text) > 0 and isinstance(input_text[0], list)):
        return input_text

    if args.ensemble_mode == "first":
        # SAFE: 5 prompts total
        return [cls_prompts[0] for cls_prompts in input_text]

    # flatten: 5*K prompts
    k = min(args.ensemble_k, min(len(x) for x in input_text))
    flat = []
    for cls_prompts in input_text:
        flat.extend(cls_prompts[:k])

    print("\n[NOTE] prompt_ensemble with 'flatten' needs model aggregation (mean over K per class).")
    print("      If GenerateModel assumes #text_prompts == #classes (5), use --ensemble-mode first.\n")
    return flat


# ==================== Training Function ====================
def run_training(args: argparse.Namespace) -> None:
    log_txt_path = os.path.join(args.output_path, "log.txt")
    log_curve_path = os.path.join(args.output_path, "log.png")
    log_confusion_matrix_path = os.path.join(args.output_path, "confusion_matrix.png")
    checkpoint_path = os.path.join(args.output_path, "model.pth")
    best_checkpoint_path = os.path.join(args.output_path, "model_best.pth")

    best_uar = 0.0
    start_epoch = 0
    recorder = RecorderMeter(args.epochs)

    # Build model + prompts
    print("=> Building model...")
    class_names, input_text = get_class_info(args)  # now uses prompt_set + text_type
    input_text = _resolve_input_text_for_ensemble(args, input_text)
    model = build_model(args, input_text).to(args.device)
    print("=> Model built and moved to device successfully.")

    # Load data
    print("=> Building dataloaders...")
    train_loader, val_loader = build_dataloaders(args)
    print("=> Dataloaders built successfully.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(
        [
            {"params": model.temporal_net.parameters(), "lr": args.lr},
            {"params": model.temporal_net_body.parameters(), "lr": args.lr},
            {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
            {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
            {"params": model.project_fc.parameters(), "lr": args.lr_image_encoder},
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    trainer = Trainer(model, criterion, optimizer, scheduler, args.device, log_txt_path)

    best_war = 0.0

    for epoch in range(start_epoch, args.epochs):
        inf = f"******************** Epoch: {epoch} ********************"
        start_time = time.time()
        print(inf)
        with open(log_txt_path, "a") as f:
            f.write(inf + "\n")

        # Log current learning rates
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_str = " ".join([f"{lr:.1e}" for lr in current_lrs])
        log_msg = f"Current learning rates: {lr_str}"
        print(log_msg)
        with open(log_txt_path, "a") as f:
            f.write(log_msg + "\n")

        # Train & Validate
        train_war, train_uar, train_los, _ = trainer.train_epoch(train_loader, epoch)
        val_war, val_uar, val_los, _ = trainer.validate(val_loader, str(epoch))

        print(f"Train set: WAR: {train_war:.4f}, UAR: {train_uar:.4f}")
        print(f"Validation set: WAR: {val_war:.4f}, UAR: {val_uar:.4f}")

        with open(log_txt_path, "a") as f:
            f.write(f"Train set: WAR: {train_war:.4f}, UAR: {train_uar:.4f}\n")
            f.write(f"Validation set: WAR: {val_war:.4f}, UAR: {val_uar:.4f}\n")

        scheduler.step()

        # Save checkpoint
        is_best = val_uar > best_uar
        best_uar = max(val_uar, best_uar)
        best_war = max(val_war, best_war)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_uar,
                "optimizer": optimizer.state_dict(),
                "recorder": recorder,
            },
            is_best,
            checkpoint_path,
            best_checkpoint_path,
        )

        # Record metrics
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_war, val_los, val_war)
        recorder.plot_curve(log_curve_path)

        print(f"The best WAR: {best_war:.4f}")
        print(f"The best UAR: {best_uar:.4f}")
        print(f"An epoch time: {epoch_time:.2f}s\n")

        with open(log_txt_path, "a") as f:
            f.write(f"The best WAR: {best_war:.4f}\n")
            f.write(f"The best UAR: {best_uar:.4f}\n")
            f.write(f"An epoch time: {epoch_time:.2f}s\n\n")

    # Final evaluation with best model
    ckpt = torch.load(best_checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    computer_uar_war(
        val_loader=val_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset}",
    )


def run_eval(args: argparse.Namespace) -> None:
    print("=> Starting evaluation mode...")
    log_txt_path = os.path.join(args.output_path, "log.txt")
    log_confusion_matrix_path = os.path.join(args.output_path, "confusion_matrix.png")

    class_names, input_text = get_class_info(args)
    input_text = _resolve_input_text_for_ensemble(args, input_text)
    model = build_model(args, input_text).to(args.device)

    if not args.eval_checkpoint:
        raise ValueError("--eval-checkpoint is required in eval mode.")

    model.load_state_dict(
        torch.load(args.eval_checkpoint, map_location=args.device, weights_only=False)["state_dict"]
    )

    _, val_loader = build_dataloaders(args)

    computer_uar_war(
        val_loader=val_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset}",
    )
    print("=> Evaluation complete.")


# ==================== Entry Point ====================
if __name__ == "__main__":
    args = parser.parse_args()
    args = setup_environment(args)
    args = setup_paths_and_logging(args)

    if args.mode == "eval":
        run_eval(args)
    else:
        run_training(args)