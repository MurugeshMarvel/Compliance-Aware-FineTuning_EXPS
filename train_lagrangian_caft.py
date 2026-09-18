"""
train_lagrangian_caft.py — Compliance-Aware Fine-Tuning with Lagrangian Constraint
===================================================================================
Fine-tunes MedGemma using a dual-objective Lagrangian optimisation:

    L = L_task + λ · (L_comp − ε)

  L_task  : standard causal-LM cross-entropy on medical instruction data
  L_comp  : masked cross-entropy on compliance alignment pairs
              (prompt tokens masked; loss only on compliant_response tokens)
  λ       : Lagrangian multiplier, updated via dual ascent
  ε       : compliance budget (max tolerated violation threshold)

Data
----
  Task dataset   : data/train/  (Arrow, alpa_care-med_instruct-52K)
                   Schema: {instruction, input, output}
  Compliance data: alignment/alignment_data.json  (300 pairs)
                   Schema: {prohibited_prompt, compliant_response, ...}

Hardware
--------
  Tested on: NVIDIA A100 / H100 (bf16, gradient checkpointing)
  Notebook  : evaluate_caft.ipynb (MPS / CUDA / CPU compatible)

Usage
-----
  uv run python train_lagrangian_caft.py
"""

import itertools
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger("caft.train")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
TASK_DATA_DIR  = PROJECT_ROOT / "data" / "train"
COMP_DATA_FILE = PROJECT_ROOT / "alignment" / "alignment_data.json"
OUTPUT_DIR     = PROJECT_ROOT / "outputs" / "medgemma-caft-adapter"

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_ID        = os.getenv("MODEL_ID", "google/medgemma-1.5-4b-it")
EPOCHS          = 3
BATCH_SIZE      = 4
LR              = 5e-5
MAX_LENGTH      = 512
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
EPSILON         = 0.05   # compliance budget: max acceptable L_comp
ETA             = 0.01   # dual ascent step size for λ
LAM_MAX         = 50.0   # hard cap on λ to prevent runaway penalty
INITIAL_LAMBDA  = 0.1
LOG_EVERY       = 10     # log every N batches
SAVE_EVERY      = 1      # save adapter every N epochs
TRAIN_SIZE      = 10_000  # task examples used for training
TEST_SIZE       = 1_000   # task examples held out for evaluation
RANDOM_SEED     = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset: Medical Instruction Task Data ─────────────────────────────────────

class MedInstructDataset(Dataset):
    """
    Loads alpa_care-med_instruct-52K from Arrow files.
    Schema: {instruction, input, output}
    Format: full prompt+response as causal-LM target (labels = input_ids).
    """

    SYSTEM_PREAMBLE = (
        "You are a compliance-aware clinical AI assistant trained to support healthcare "
        "professionals and patients with accurate, evidence-based medical information. "
        "You are explicitly designed to uphold regulatory and ethical standards including "
        "HIPAA, GDPR, ICH-GCP, and FDA guidelines. "
        "You must never disclose protected health information, recommend off-label treatments "
        "without appropriate clinical justification, provide definitive diagnoses without "
        "examination, or assist with actions that violate patient privacy, informed consent, "
        "or clinical trial integrity. "
        "When a request may cause regulatory harm or patient safety risk, you must decline "
        "and redirect to a licensed professional or official reporting channel.\n\n"
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
        rec = self.records[idx]
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
            "labels":         input_ids.clone(),  # full sequence is target
        }


# ── Dataset: Compliance Alignment Data ────────────────────────────────────────

class ComplianceAlignmentDataset(Dataset):
    """
    Loads alignment_data.json (300 pairs).
    Schema: {prohibited_prompt, compliant_response, ...}
    Labels: prompt tokens masked with -100; loss computed only on response tokens.
    """

    def __init__(self, tokenizer, max_length: int = MAX_LENGTH):
        logger.info(f"Loading compliance dataset from {COMP_DATA_FILE}")
        if not COMP_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Compliance data not found: {COMP_DATA_FILE}\n"
                "Run: python alignment/generate_alignment_data.py"
            )
        with open(COMP_DATA_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        if not raw:
            raise ValueError(
                f"alignment_data.json is empty. "
                "Run: python alignment/generate_alignment_data.py"
            )
        self.records   = raw
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Compliance dataset: {len(self.records):,} pairs")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec    = self.records[idx]
        prompt = rec["prohibited_prompt"]
        resp   = rec["compliant_response"]

        # Tokenise prompt alone to know where response starts
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length // 2,  # reserve half for response
            add_special_tokens=True,
        )
        prompt_len = len(prompt_enc["input_ids"])

        # Tokenise full sequence: prompt + "\n" + response
        full_text = prompt + "\n" + resp
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Mask prompt tokens: model should only be penalised on response tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100                        # mask prompt
        labels[attention_mask == 0] = -100                # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Compliance Loss ────────────────────────────────────────────────────────────

def compute_compliance_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Masked cross-entropy: penalises the model only on compliant_response tokens.
    Higher value = model is failing to generate the compliance-aligned response.

    logits : (B, T, V)
    labels : (B, T)  — prompt tokens are -100 (ignored)
    """
    shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = labels[..., 1:].contiguous()        # (B, T-1)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ── Main Training Loop ─────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token   # Gemma has no default pad token

    # ── Model + LoRA ───────────────────────────────────────────────────────
    logger.info(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",           # A100 / multi-GPU; not used on MPS
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

    task_ds  = MedInstructDataset(train_records, tokenizer)
    test_ds  = MedInstructDataset(test_records,  tokenizer)
    comp_ds  = ComplianceAlignmentDataset(tokenizer)

    train_loader = DataLoader(task_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    comp_loader  = DataLoader(comp_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR)

    # ── Lagrangian multiplier λ ─────────────────────────────────────────────
    # Updated via manual dual ascent — NOT via autograd / optimizer.
    lam = torch.tensor([INITIAL_LAMBDA], dtype=torch.float32, device=DEVICE)

    # ── Training ───────────────────────────────────────────────────────────
    logger.info("Starting training")
    logger.info(f"  epochs={EPOCHS} | batch={BATCH_SIZE} | lr={LR} | ε={EPSILON} | η={ETA}")

    model.train()

    for epoch in range(1, EPOCHS + 1):
        comp_iter = itertools.cycle(comp_loader)   # cycle: 300 pairs << 52K task examples

        for batch_idx, task_batch in enumerate(train_loader):
            optimizer.zero_grad()

            # ── PART A: Task loss ──────────────────────────────────────────
            task_inputs = {k: v.to(DEVICE) for k, v in task_batch.items()}
            task_out    = model(**task_inputs)
            l_task      = task_out.loss

            # ── PART B: Compliance loss ────────────────────────────────────
            comp_batch  = next(comp_iter)
            comp_inputs = {k: v.to(DEVICE) for k, v in comp_batch.items()}
            # NOTE: NO torch.no_grad() here — gradients must flow back through
            # l_comp for the primal update to penalise non-compliant outputs.
            comp_out = model(
                input_ids=comp_inputs["input_ids"],
                attention_mask=comp_inputs["attention_mask"],
            )
            l_comp = compute_compliance_loss(comp_out.logits, comp_inputs["labels"])

            # ── PART C: Lagrangian primal objective ────────────────────────
            # lam.detach() keeps λ out of the model's gradient graph.
            # The model is updated to minimise task loss + penalised violation.
            total_loss = l_task + lam.detach().item() * (l_comp - EPSILON)
            total_loss.backward()
            optimizer.step()

            # ── PART D: Dual ascent — update λ ────────────────────────────
            # λ ← max(0, λ + η · (L_comp − ε))
            # When violation > 0 (non-compliant), λ increases → higher penalty next step.
            with torch.no_grad():
                violation  = l_comp.item() - EPSILON
                lam.data   = torch.clamp(lam.data + ETA * violation, min=0.0, max=LAM_MAX)

            if batch_idx % LOG_EVERY == 0:
                logger.info(
                    f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx:>5} | "
                    f"L_task={l_task.item():.4f} | "
                    f"L_comp={l_comp.item():.4f} | "
                    f"violation={violation:+.4f} | "
                    f"λ={lam.item():.4f}"
                )

        # ── Eval on held-out test split ────────────────────────────────────
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_batch in test_loader:
                test_inputs = {k: v.to(DEVICE) for k, v in test_batch.items()}
                test_outputs = model(**test_inputs)
                test_loss += test_outputs.loss.item()
        avg_test_loss = test_loss / len(test_loader)
        logger.info(f"Epoch {epoch} — test  L_task={avg_test_loss:.4f}  λ={lam.item():.4f}")
        model.train()

        # ── Save adapter checkpoint per epoch ─────────────────────────────
        if epoch % SAVE_EVERY == 0:
            ckpt_path = OUTPUT_DIR / f"epoch-{epoch}"
            model.save_pretrained(str(ckpt_path))
            logger.info(f"Checkpoint saved → {ckpt_path}")

    # ── Save final adapter ─────────────────────────────────────────────────
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info(f"Training complete. Final adapter saved → {OUTPUT_DIR}")
    logger.info(f"Final λ = {lam.item():.6f}")


if __name__ == "__main__":
    main()
