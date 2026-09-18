"""
train_baseline_sft.py — Standard Supervised Fine-Tuning (No Compliance Constraint)
====================================================================================
Fine-tunes MedGemma on the medical instruction dataset ONLY.

This is the BASELINE (step 5 in the experiment pipeline):
  Fine-tune(MedGemma + train set) → expected compliance eraser effect:
  compliance score should DROP below the base model's ~60%.

Objective (pure task loss only):
    L = L_task   (standard causal-LM cross-entropy)

No Lagrangian multiplier. No compliance data. No dual ascent.

Data
----
  Task dataset : data/train/  (Arrow, alpa_care-med_instruct-52K)
                 Schema: {instruction, input, output}

Hardware
--------
  Tested on: NVIDIA A100 / H100 (bf16, gradient checkpointing)

Usage
-----
  uv run python train_baseline_sft.py
"""

import logging
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger("baseline.train")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
TASK_DATA_DIR = PROJECT_ROOT / "data" / "train"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "medgemma-baseline-adapter"

# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_ID       = os.getenv("MODEL_ID", "google/medgemma-1.5-4b-it")
EPOCHS         = 3
BATCH_SIZE     = 4
LR             = 5e-5
MAX_LENGTH     = 512
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LOG_EVERY      = 10   # log every N batches
SAVE_EVERY     = 1    # save checkpoint every N epochs
TRAIN_SIZE     = 10_000   # examples used for training
TEST_SIZE      = 1_000    # examples held out for evaluation
RANDOM_SEED    = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ────────────────────────────────────────────────────────────────────

class MedInstructDataset(Dataset):
    """
    Loads alpa_care-med_instruct-52K from Arrow files.
    Schema: {instruction, input, output}
    Full sequence tokenized as causal-LM target (labels = input_ids).
    """

    SYSTEM_PREAMBLE = (
        "You are a clinical AI assistant trained to support healthcare professionals "
        "and patients with accurate, evidence-based medical information. "
        "Always prioritise patient safety, regulatory compliance, and clinical best practice.\n\n"
    )

    TEMPLATE_WITH_INPUT = (
        "{system}"
        "### Clinical Question:\n{instruction}\n\n"
        "### Patient / User Context:\n{input}\n\n"
        "### Clinical Answer:\n{output}"
    )
    TEMPLATE_NO_INPUT = (
        "{system}"
        "### Clinical Question:\n{instruction}\n\n"
        "### Clinical Answer:\n{output}"
    )

    def __init__(self, records: list, tokenizer, max_length: int = MAX_LENGTH):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Dataset split: {len(self.records):,} examples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec         = self.records[idx]
        instruction = rec.get("instruction", "")
        inp         = rec.get("input", "")
        output      = rec.get("output", "")

        if inp.strip():
            text = self.TEMPLATE_WITH_INPUT.format(
                system=self.SYSTEM_PREAMBLE,
                instruction=instruction,
                input=inp,
                output=output,
            )
        else:
            text = self.TEMPLATE_NO_INPUT.format(
                system=self.SYSTEM_PREAMBLE,
                instruction=instruction,
                output=output,
            )

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         input_ids.clone(),   # full sequence is target
        }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Model + LoRA ───────────────────────────────────────────────────────
    logger.info(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Dataset split (10K train / 1K test) ──────────────────────────────
    import random
    logger.info(f"Loading full dataset from {TASK_DATA_DIR}")
    all_records = list(load_from_disk(str(TASK_DATA_DIR)))
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)
    train_records = all_records[:TRAIN_SIZE]
    test_records  = all_records[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]
    logger.info(f"Split: {len(train_records):,} train  |  {len(test_records):,} test")

    train_ds = MedInstructDataset(train_records, tokenizer)
    test_ds  = MedInstructDataset(test_records,  tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR)

    # ── Training ───────────────────────────────────────────────────────────
    logger.info("Starting baseline SFT (no compliance constraint)")
    logger.info(f"  epochs={EPOCHS} | batch={BATCH_SIZE} | lr={LR}")

    model.train()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            inputs  = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)
            loss    = outputs.loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % LOG_EVERY == 0:
                logger.info(
                    f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx:>5} | "
                    f"L_task={loss.item():.4f}"
                )

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} complete — avg train L_task={avg_train_loss:.4f}")

        # ── Eval on held-out test split ────────────────────────────────────
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_batch in test_loader:
                test_inputs = {k: v.to(DEVICE) for k, v in test_batch.items()}
                test_outputs = model(**test_inputs)
                test_loss += test_outputs.loss.item()
        avg_test_loss = test_loss / len(test_loader)
        logger.info(f"Epoch {epoch} — test  L_task={avg_test_loss:.4f}")
        model.train()

        if epoch % SAVE_EVERY == 0:
            ckpt_path = OUTPUT_DIR / f"epoch-{epoch}"
            model.save_pretrained(str(ckpt_path))
            logger.info(f"Checkpoint saved → {ckpt_path}")

    # ── Save final adapter ─────────────────────────────────────────────────
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info(f"Baseline SFT complete. Adapter saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
