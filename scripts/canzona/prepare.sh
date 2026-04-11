#!/bin/bash
# ============================================================
# Megatron-LM WikiText-103 Data Preprocessing Script
# Task: Pretraining with Next Token Prediction (NTP)
# Data source: HuggingFace datasets
# ============================================================

set -e  # Exit immediately if any command fails

# ============================================================
# Configuration (modify as needed)
# ============================================================
MEGATRON_ROOT="../../"
DATA_DIR="./data"
OUTPUT_PREFIX="${DATA_DIR}/wikitext103"
WORKERS=8

# ============================================================
# Utility functions
# ============================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_success() {
    if [ $? -eq 0 ]; then
        log "✅ $1 succeeded"
    else
        log "❌ $1 failed, exiting"
        exit 1
    fi
}

# ============================================================
# Step 0: Initialize directories
# ============================================================
log "========== Step 0: Initialize directories =========="
mkdir -p ${DATA_DIR}/vocab
check_success "Directory initialization"

# ============================================================
# Step 1: Install dependencies
# ============================================================
log "========== Step 1: Install dependencies =========="
pip install datasets
check_success "datasets installation"

# ============================================================
# Step 2: Download vocabulary files from HuggingFace
# ============================================================
log "========== Step 2: Download vocabulary files =========="

if [ -f "${DATA_DIR}/vocab/gpt2-vocab.json" ] && [ -f "${DATA_DIR}/vocab/gpt2-merges.txt" ]; then
    log "⏭️  Vocabulary files already exist, skipping"
else
    DATA_DIR=${DATA_DIR} python3 << 'PYEOF'
import os
from transformers import GPT2Tokenizer

data_dir = os.environ.get('DATA_DIR', './data')
vocab_dir = os.path.join(data_dir, 'vocab')

print("Downloading GPT2 tokenizer files from HuggingFace...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.save_pretrained(vocab_dir)

# Rename to match Megatron expected filenames
import shutil
import os

src_vocab = os.path.join(vocab_dir, 'vocab.json')
src_merge = os.path.join(vocab_dir, 'merges.txt')
dst_vocab = os.path.join(vocab_dir, 'gpt2-vocab.json')
dst_merge = os.path.join(vocab_dir, 'gpt2-merges.txt')

if os.path.exists(src_vocab):
    shutil.copy(src_vocab, dst_vocab)
    print(f"Saved: {dst_vocab}")

if os.path.exists(src_merge):
    shutil.copy(src_merge, dst_merge)
    print(f"Saved: {dst_merge}")

print("Vocabulary files ready!")
PYEOF
    check_success "Vocabulary files download"
fi

# ============================================================
# Step 3: Download WikiText-103 from HuggingFace and
#         convert to JSON format
#
# Each document is stored as one JSON line.
# Megatron will concatenate all documents during training.
# ============================================================
log "========== Step 3: Download and convert WikiText-103 =========="

if [ -f "${DATA_DIR}/wikitext103_train.json" ]; then
    log "⏭️  wikitext103_train.json already exists, skipping"
else
    DATA_DIR=${DATA_DIR} python3 << 'PYEOF'
import json
import os
from datasets import load_dataset

data_dir = os.environ.get('DATA_DIR', './data')

print("Downloading WikiText-103 from HuggingFace...")
# wikitext-103-raw-v1: raw text without preprocessing
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

def save_split_to_json(split_data, output_path):
    """
    Save a dataset split to JSON format (one document per line).
    Empty lines and whitespace-only lines are skipped.
    """
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in split_data:
            text = sample['text'].strip()
            if not text:
                continue
            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            count += 1
    print(f"  -> Written {count} documents to {output_path}")
    return count

# Save all splits
splits = ['train', 'validation', 'test']
for split in splits:
    # validation -> valid for naming consistency
    split_name = 'valid' if split == 'validation' else split
    output_path = os.path.join(data_dir, f'wikitext103_{split_name}.json')
    print(f"Processing {split} split...")
    save_split_to_json(dataset[split], output_path)

print("All splits converted successfully!")
PYEOF
    check_success "WikiText-103 download and conversion"
fi

# ============================================================
# Step 4: Convert to Megatron binary format
#
# Key NTP-related flags:
#   --append-eod : Appends <EOD> token at end of each document.
#                  During training, Megatron concatenates all
#                  documents into a single stream and splits by
#                  seq-length, which is exactly what NTP requires.
#
# Note: We only preprocess the train split here.
#       Megatron handles train/val/test splitting via --split flag.
# ============================================================
log "========== Step 4: Convert to Megatron binary format =========="

if [ -f "${OUTPUT_PREFIX}_text_document.bin" ]; then
    log "⏭️  Megatron binary format already exists, skipping"
else
    python ${MEGATRON_ROOT}/tools/preprocess_data.py \
        --input ${DATA_DIR}/wikitext103_train.json \
        --output-prefix ${OUTPUT_PREFIX} \
        --vocab-file ${DATA_DIR}/vocab/gpt2-vocab.json \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file ${DATA_DIR}/vocab/gpt2-merges.txt \
        --append-eod \
        --workers ${WORKERS}
    check_success "Megatron binary format conversion"
fi

# ============================================================
# Done! Print summary
# ============================================================
log "============================================"
log "🎉 All preprocessing steps completed!"
log "============================================"
log ""
log "Output files:"
ls -lh ${OUTPUT_PREFIX}_text_document.* 2>/dev/null
log ""
log "Dataset size estimation:"
DATA_DIR=${DATA_DIR} python3 << 'PYEOF'
import os

data_dir = os.environ.get('DATA_DIR', './data')

# Count total lines (documents) in train json
train_json = os.path.join(data_dir, 'wikitext103_train.json')
with open(train_json, 'r') as f:
    num_docs = sum(1 for _ in f)

# Estimate total tokens (rough estimate: ~5 chars per token)
total_chars = os.path.getsize(train_json)
est_tokens  = total_chars // 5

SEQ_LEN           = 4096
GLOBAL_BATCH_SIZE = 1024

est_samples = est_tokens // SEQ_LEN
est_steps   = est_samples // GLOBAL_BATCH_SIZE

print(f"  Estimated total tokens  : {est_tokens:,}")
print(f"  Estimated total samples : {est_samples:,}  (seq_len={SEQ_LEN})")
print(f"  Suggested TRAIN_STEPS   : {est_steps:,}  (1 epoch, GBS={GLOBAL_BATCH_SIZE})")
print(f"  Suggested TRAIN_STEPS   : {est_steps*3:,}  (3 epochs, GBS={GLOBAL_BATCH_SIZE})")
PYEOF
log ""
log "Use the following arguments for training:"
log "  --data-path ${OUTPUT_PREFIX}_text_document"
log "  --vocab-file ${DATA_DIR}/vocab/gpt2-vocab.json"
log "  --merge-file ${DATA_DIR}/vocab/gpt2-merges.txt"
log "  --split 949,50,1"