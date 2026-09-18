# Compliance-Aware Fine-Tuning — live notebooks

Companion code for the Tri-Valley Tech Meetup talk on what standard fine-tuning
does to a clinical model's compliance behaviour, and how to stop it.

Four notebooks, run in order. Click a badge and it opens in Google Colab — no
download, no setup, no paths to edit.

| | Notebook | What it does | GPU | HF token |
|---|---|---|---|---|
| 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/caft-compliance-talk/blob/main/Use-Case.ipynb) `Use-Case.ipynb` | The task data, the three-dataset split, what an adversarial compliance probe looks like | no | no |
| 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/caft-compliance-talk/blob/main/Standard_Finetuning.ipynb) `Standard_Finetuning.ipynb` | Fine-tune the ordinary way, then audit what it cost | yes | yes |
| 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/caft-compliance-talk/blob/main/CAFT_Finetuning.ipynb) `CAFT_Finetuning.ipynb` | Put the rule inside the loss with a Lagrangian constraint | yes | yes |
| 4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/caft-compliance-talk/blob/main/Comparing_Results.ipynb) `Comparing_Results.ipynb` | Three models, 72 probes, and the guardrail experiment | no | no |

Notebooks 1 and 4 run on anything, including a laptop with no GPU. They read
results that are already in this repo, so they work even if the wifi is bad.

---

## Running notebooks 2 and 3

These two actually train, so they need two things.

**A GPU.** In Colab: `Runtime` → `Change runtime type` → `T4 GPU`. The free tier
is enough for the demo configuration (`DEMO_MODE = True`, 40 steps, 3–4 minutes).

**Access to MedGemma,** which is a gated model. Three one-time steps:

1. Sign in to Hugging Face and accept the licence at
   [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it).
   This is the step people miss — a token alone is not enough.
2. Create a **read** token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. In Colab, click the **key icon** in the left sidebar and add a secret named
   `HF_TOKEN` with that value, then toggle notebook access on. Running locally,
   put `HF_TOKEN=hf_...` in a `.env` file next to the notebook instead.

**If you cannot get through the gate,** the notebooks still run. The setup cell
detects the problem and swaps in `Qwen/Qwen2.5-0.5B-Instruct`, which is small
and ungated. The LoRA config, the label masking, the training loop and the
Lagrangian constraint are all identical — only the weights differ. You will see
the method work; you will not reproduce the numbers from the talk, because that
model is a fraction of the size and has no medical post-training.

---

## Running it locally instead

```bash
git clone https://github.com/MurugeshMarvel/caft-compliance-talk.git
cd caft-compliance-talk
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The setup cell finds the repo by walking up from wherever the notebook sits, so
there is nothing to configure.

---

## What the setup cell actually does

Every notebook opens with the same short bootstrap that clones this repo if it
is not already on disk, then calls `caft_colab.setup()`. That function is one
readable file, [`caft_colab.py`](caft_colab.py), and it does five things:

1. installs dependencies (on Colab only — a local environment is left alone)
2. resolves the project root and checks the data files are present
3. finds a Hugging Face token: Colab secret → environment variable → `.env` → prompt
4. checks whether your token can actually reach gated MedGemma, and falls back
   to the ungated model if not
5. picks a device and a dtype that will not blow up — bfloat16 on Ampere and
   newer, float16 on a T4, float32 on CPU

It prints a status line for each step, so when something is wrong you can see
which of the five it was.

---

## Data

| What | Where it comes from | Size |
|---|---|---|
| Task data — AlpaCare-MedInstruct-52k | pulled from the Hub at runtime, public and ungated | ~37 MB, cached |
| Alignment set `D_A` — 1,475 prohibited-prompt → compliant-response pairs | `caft_exp_scripts/alignment/alignment_data.json` | in this repo |
| Audit set `D_R` — 72 adversarial probes, held out | `caft_exp_scripts/audit/audit_data.json` | in this repo |
| Pre-computed run results for all three models | `caft_exp_scripts/eval/results/` | in this repo |

The audit set is never trained on, by either the baseline or CAFT. That is the
rule that makes the compliance score mean anything: train on your audit probes
and you are measuring memorisation, not safety.

---

## Reproducing the full run

The notebooks are the demo configuration. The published numbers come from the
scripts, at 3 epochs over 10,000 task examples with `max_length=512` on an A100:

```bash
python caft_exp_scripts/train_baseline_sft.py       # standard SFT baseline
python caft_exp_scripts/train_lagrangian_caft.py    # CAFT
python caft_exp_scripts/eval/run_inference.py       # 72 probes x 3 models
python caft_exp_scripts/eval/run_evaluation.py      # judge scoring
```

Hyperparameters are identical across both training scripts except for the
constraint: LoRA `r=16`, `alpha=32`, dropout `0.05`, targets `q_proj`/`v_proj`,
`lr=5e-5`, batch 4, seed 42. CAFT adds `ε=0.05`, `η=0.01`, `λ₀=0.1`, `λ_max=50`.

---

## Licence and credit

Code in this repository is released for teaching purposes alongside the talk.
MedGemma is Google's, under its own licence, which you accept on the model page.
AlpaCare-MedInstruct-52k is published by lavita on the Hugging Face Hub.
