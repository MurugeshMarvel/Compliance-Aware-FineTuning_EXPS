from torch.utils.data import DataLoader, Dataset
import logging
import sys

logging.basicConfig(
    level=logging.INFO,                                      # Set your desired log level (DEBUG, INFO, etc.)
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", # Clean text layout
    datefmt="%H:%M:%S",                                      # Short time format for easy scanning
    stream=sys.stdout                                        # Force output to stdout to avoid red shading
)
logger = logging.getLogger("Data_Utils")

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

    def __init__(self, records: list, tokenizer, max_length: int):
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
