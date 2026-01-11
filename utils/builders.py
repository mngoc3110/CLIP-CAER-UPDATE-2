# builders.py

import argparse
from typing import Tuple, List, Union
import os
import torch
import torch.utils.data
from clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *  # noqa: import your prompt definitions


PromptType = Union[List[str], List[List[str]]]


# ----------------------------
# Prompt getters (robust)
# ----------------------------
def _get_from_globals(name: str, default=None):
    return globals().get(name, default)


def _resolve_prompt_bank():
    """
    Prefer the merged dict-style prompt banks if they exist:
      - BASELINE
      - STUDENT_CONTEXT
      - DESCRIPTORS
      - PROMPT_ENSEMBLE_5
    Otherwise fallback to legacy single variables.
    """
    baseline = _get_from_globals("BASELINE", None)
    student = _get_from_globals("STUDENT_CONTEXT", None)
    descriptors = _get_from_globals("DESCRIPTORS", None)
    ensemble = _get_from_globals("PROMPT_ENSEMBLE_5", None)

    return baseline, student, descriptors, ensemble


def _print_input_text(input_text: PromptType) -> None:
    print("\nInput Text Prompts:")
    if not input_text:
        print("(empty)")
        return

    if isinstance(input_text[0], list):
        # ensemble list[list[str]]
        for ci, class_prompts in enumerate(input_text):
            print(f"- Class {ci}:")
            for pj, p in enumerate(class_prompts):
                print(f"   [{pj}] {p}")
    else:
        for i, text in enumerate(input_text):
            print(f"[{i}] {text}")


# ----------------------------
# Model builder
# ----------------------------
def build_model(args: argparse.Namespace, input_text: PromptType) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    CLIP_model, _ = clip.load(args.clip_path, device="cpu")

    _print_input_text(input_text)

    print("\nInstantiating GenerateModel...")
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    # Freeze everything first
    for _, p in model.named_parameters():
        p.requires_grad = False

    # Unfreeze CLIP image encoder if desired
    if getattr(args, "lr_image_encoder", 0.0) > 0:
        for name, p in model.named_parameters():
            if "image_encoder" in name:
                p.requires_grad = True

    # Unfreeze your trainable modules
    trainable_keywords = [
        "temporal_net",
        "prompt_learner",
        "temporal_net_body",
        "project_fc",
        "face_adapter",
    ]

    print("\nTrainable parameters:")
    for name, p in model.named_parameters():
        if any(k in name for k in trainable_keywords):
            p.requires_grad = True
            print(f"- {name}")
    print("************************\n")

    return model


# ----------------------------
# Class info / prompt selection
# ----------------------------
def get_class_info(args: argparse.Namespace) -> Tuple[List[str], PromptType]:
    """
    Returns:
      class_names: fixed labels for metrics/confusion matrix
      input_text : prompt list or prompt ensemble list[list[str]]

    Uses:
      args.prompt_set in {baseline, student, descriptor_face, descriptor_body, descriptor_full, ensemble}
      args.text_type in {class_names, class_names_with_context, class_descriptor, class_descriptor_only_face,
                         class_descriptor_only_body, prompt_ensemble}
    """
    if args.dataset != "RAER":
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented yet.")

    # Clean labels (avoid punctuation mismatch in confusion matrix)
    class_names = ["Neutrality", "Enjoyment", "Confusion", "Fatigue", "Distraction"]

    baseline, student, descriptors, ensemble_bank = _resolve_prompt_bank()

    prompt_set = getattr(args, "prompt_set", "baseline")
    text_type = getattr(args, "text_type", "class_descriptor")

    # ---------
    # 1) Choose a prompt source (prompt_set)
    # ---------
    # We will map prompt_set -> (class_names_5, class_names_with_context_5, desc_face, desc_body, desc_full, ensemble)
    # using dict banks if available, else fallback to legacy variables.

    if prompt_set == "baseline":
        # dict bank
        if baseline is not None:
            cn = baseline.get("class_names_5", class_names)
            cnc = baseline.get("class_names_with_context_5", _get_from_globals("class_names_with_context_5", cn))
            desc_face = baseline.get("class_descriptor_5_only_face", _get_from_globals("class_descriptor_5_only_face", None))
            desc_body = baseline.get("class_descriptor_5_only_body", _get_from_globals("class_descriptor_5_only_body", None))
            desc_full = baseline.get("class_descriptor_5", _get_from_globals("class_descriptor_5", None))
        else:
            # legacy
            cn = _get_from_globals("class_names_5", class_names)
            cnc = _get_from_globals("class_names_with_context_5", cn)
            desc_face = _get_from_globals("class_descriptor_5_only_face", None)
            desc_body = _get_from_globals("class_descriptor_5_only_body", None)
            desc_full = _get_from_globals("class_descriptor_5", None)

        ens = _get_from_globals("prompt_ensemble_5", ensemble_bank)

    elif prompt_set == "student":
        # student dict bank
        if student is not None:
            cn = student.get("class_names_5", class_names)
            cnc = student.get("class_names_with_context_5", _get_from_globals("class_names_with_context_5", cn))
            # student may or may not define descriptors; fallback to baseline/legacy
            desc_face = _get_from_globals("class_descriptor_5_only_face", None)
            desc_body = _get_from_globals("class_descriptor_5_only_body", None)
            desc_full = _get_from_globals("class_descriptor_5", None)
        else:
            # if no student bank, fallback to legacy "student" variables if you defined them
            cn = _get_from_globals("class_names_5_student", _get_from_globals("class_names_5", class_names))
            cnc = _get_from_globals("class_names_with_context_5_student", _get_from_globals("class_names_with_context_5", cn))
            desc_face = _get_from_globals("class_descriptor_5_only_face_student", _get_from_globals("class_descriptor_5_only_face", None))
            desc_body = _get_from_globals("class_descriptor_5_only_body_student", _get_from_globals("class_descriptor_5_only_body", None))
            desc_full = _get_from_globals("class_descriptor_5_student", _get_from_globals("class_descriptor_5", None))

        ens = _get_from_globals("prompt_ensemble_5_student", _get_from_globals("prompt_ensemble_5", ensemble_bank))

    elif prompt_set in ["descriptor_face", "descriptor_body", "descriptor_full"]:
        # descriptor dict bank: DESCRIPTORS = {"only_face": [...], "only_body": [...], "face_and_body": [...]}
        if descriptors is not None:
            cn = _get_from_globals("class_names_5", class_names)
            cnc = _get_from_globals("class_names_with_context_5", cn)
            desc_face = descriptors.get("only_face", _get_from_globals("class_descriptor_5_only_face", None))
            desc_body = descriptors.get("only_body", _get_from_globals("class_descriptor_5_only_body", None))
            desc_full = descriptors.get("face_and_body", _get_from_globals("class_descriptor_5", None))
        else:
            # fallback legacy
            cn = _get_from_globals("class_names_5", class_names)
            cnc = _get_from_globals("class_names_with_context_5", cn)
            desc_face = _get_from_globals("class_descriptor_5_only_face", None)
            desc_body = _get_from_globals("class_descriptor_5_only_body", None)
            desc_full = _get_from_globals("class_descriptor_5", None)

        ens = _get_from_globals("prompt_ensemble_5", ensemble_bank)

    elif prompt_set == "ensemble":
        cn = _get_from_globals("class_names_5", class_names)
        cnc = _get_from_globals("class_names_with_context_5", cn)
        desc_face = _get_from_globals("class_descriptor_5_only_face", None)
        desc_body = _get_from_globals("class_descriptor_5_only_body", None)
        desc_full = _get_from_globals("class_descriptor_5", None)
        ens = _get_from_globals("prompt_ensemble_5", ensemble_bank)

    else:
        raise ValueError(f"Unknown prompt_set: {prompt_set}")

    # ---------
    # 2) Choose input_text (text_type)
    # ---------
    if text_type == "class_names":
        input_text: PromptType = cn

    elif text_type == "class_names_with_context":
        input_text = cnc

    elif text_type == "class_descriptor":
        # descriptor_full depending on prompt_set
        if prompt_set == "descriptor_face":
            input_text = desc_face if desc_face is not None else desc_full
        elif prompt_set == "descriptor_body":
            input_text = desc_body if desc_body is not None else desc_full
        elif prompt_set == "descriptor_full":
            input_text = desc_full
        else:
            input_text = desc_full

    elif text_type == "class_descriptor_only_face":
        input_text = desc_face if desc_face is not None else desc_full

    elif text_type == "class_descriptor_only_body":
        input_text = desc_body if desc_body is not None else desc_full

    elif text_type == "prompt_ensemble":
        if ens is None:
            raise ValueError("prompt_ensemble_5 not found in models/Text.py")
        input_text = ens  # list[list[str]]

    else:
        raise ValueError(f"Unknown text_type: {text_type}")

    if input_text is None:
        raise ValueError(
            f"Resolved input_text is None. Check your models/Text.py for missing variables "
            f"(prompt_set={prompt_set}, text_type={text_type})."
        )

    return class_names, input_text


# ----------------------------
# Dataloaders
# ----------------------------
def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_annotation_file_path = os.path.join(args.root_dir, args.train_annotation)

    # Prefer val_annotation if provided; else fallback to test_annotation
    val_anno = getattr(args, "val_annotation", "")
    if val_anno is None or str(val_anno).strip() == "":
        val_annotation_file_path = os.path.join(args.root_dir, args.test_annotation)
    else:
        val_annotation_file_path = os.path.join(args.root_dir, args.val_annotation)

    print("Loading train data...")
    train_data = train_data_loader(
        list_file=train_annotation_file_path,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body,
    )

    print("Loading val/test data...")
    val_data = test_data_loader(
        list_file=val_annotation_file_path,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body,
    )

    print("Creating DataLoader instances...")
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader