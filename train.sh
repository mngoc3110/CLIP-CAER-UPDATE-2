#!/bin/bash
set -e

COMMON_ARGS="\
--mode train \
--gpu mps \
--epochs 20 \
--batch-size 8 \
--lr 0.003 \
--lr-image-encoder 1e-5 \
--lr-prompt-learner 0.001 \
--weight-decay 0.0001 \
--momentum 0.9 \
--milestones 10 15 \
--gamma 0.1 \
--temporal-layers 1 \
--num-segments 16 \
--duration 1 \
--image-size 224 \
--seed 42 \
--print-freq 10 \
--root-dir ./ \
--train-annotation /content/drive/MyDrive/khoaluan/DatasetRAER/annotation/train.txt \
--test-annotation /content/drive/MyDrive/khoaluan/DatasetRAER/annotation/test.txt \
--clip-path ViT-B/32 \
--bounding-box-face /content/drive/MyDrive/khoaluan/DatasetRAER/bounding_box/face.json \
--bounding-box-body /content/drive/MyDrive/khoaluan/DatasetRAER/bounding_box/body.json \
--contexts-number 8 \
--class-token-position end \
--class-specific-contexts True \
--load_and_tune_prompt_learner True \
"

run () {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
  python main.py $COMMON_ARGS $2
}

# =========================
# 1) BASELINE (4)
# =========================
run "1/10 baseline + class_names" "\
--exper-name BL_class_names \
--prompt_set baseline \
--text-type class_names \
"

run "2/10 baseline + class_names_with_context" "\
--exper-name BL_class_names_with_context \
--prompt_set baseline \
--text-type class_names_with_context \
"

run "3/10 baseline + class_descriptor_only_face" "\
--exper-name BL_desc_only_face \
--prompt_set baseline \
--text-type class_descriptor_only_face \
"

run "4/10 baseline + class_descriptor (full)" "\
--exper-name BL_desc_full \
--prompt_set baseline \
--text-type class_descriptor \
"

# =========================
# 2) STUDENT_CONTEXT (2)
# =========================
run "5/10 student_context + class_names" "\
--exper-name ST_class_names \
--prompt_set student_context \
--text-type class_names \
"

run "6/10 student_context + class_names_with_context" "\
--exper-name ST_class_names_with_context \
--prompt_set student_context \
--text-type class_names_with_context \
"

# =========================
# 3) DESCRIPTOR (3)
# =========================
run "7/10 descriptor + only_face" "\
--exper-name DS_only_face \
--prompt_set descriptor \
--text-type class_descriptor_only_face \
"

run "8/10 descriptor + only_body" "\
--exper-name DS_only_body \
--prompt_set descriptor \
--text-type class_descriptor_only_body \
"

run "9/10 descriptor + face_and_body" "\
--exper-name DS_face_and_body \
--prompt_set descriptor \
--text-type class_descriptor \
"

# =========================
# 4) ENSEMBLE (1)
# =========================
run "10/10 ensemble" "\
--exper-name ENS_prompt_ensemble \
--prompt_set ensemble \
--text-type prompt_ensemble \
"

echo ""
echo "✅ DONE: Finished 10 jobs."