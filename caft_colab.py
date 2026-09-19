"""
caft_colab — one setup call that makes the four CAFT notebooks run anywhere.

Every notebook starts with the same short bootstrap cell, which clones this
repo if it is not already on disk and then calls:

    env = caft_colab.setup(need_gpu=False)      # notebooks 1 and 4
    env = caft_colab.setup(need_gpu=True)       # notebooks 2 and 3

`setup()` does five things, in order, and prints a status line for each:

  1. installs the Python dependencies (Colab only — a local env is left alone)
  2. downloads the study data from the Hugging Face Hub and checks it is complete
  3. finds a Hugging Face token: Colab secret -> env var -> .env -> prompt
  4. checks whether you actually have access to gated MedGemma, and falls back
     to a small ungated model if you do not, so the notebook still runs
  5. picks a device and a dtype that will not blow up

It returns a plain object whose attributes the notebooks use directly:

    env.PROJECT_ROOT  env.DATA_DIR  env.ALIGN_JSON  env.AUDIT_JSON  env.RESULTS
    env.MODEL_ID      env.HF_TOKEN  env.IS_FALLBACK
    env.DEVICE        env.DTYPE     env.IN_COLAB

Nothing here is specific to the talk — it is all plumbing, deliberately kept
in one file so an attendee can read it in two minutes and see there is no magic.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# ── The two model IDs ─────────────────────────────────────────────────────────
# MedGemma is gated: you must accept the licence on the model page while signed
# in to Hugging Face before any token will work.
MEDGEMMA_ID = "google/medgemma-1.5-4b-it"

# Ungated stand-in so the training loop still demonstrates for anyone who cannot
# get through the gate in the room. Same architecture family for our purposes:
# it is a decoder-only causal LM with q_proj / v_proj attention projections, so
# the identical LoRA config and the identical CAFT loss apply unchanged.
# It is much smaller and NOT medically post-trained — the mechanics are real,
# the numbers are not the ones in the talk.
FALLBACK_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Where the study data lives ────────────────────────────────────────────────
# A public, ungated dataset repo on the Hub holding the seven files the
# notebooks read. Keeping it off GitHub means the code repo stays small and the
# data has one home, versioned independently of the notebooks.
#
#   alignment_data.json              D_A  — 1,475 prohibited-prompt pairs
#   audit_data.json                  D_R  — 72 held-out adversarial probes
#   results/base_responses.json      pre-computed run output, base model
#   results/sft_responses.json       ..., standard SFT
#   results/caft_responses.json      ..., CAFT
#   results/judge_scores.json        the 0-5 regulatory rubric scores
#   results/utility_judge_summary.json   task-quality check, n=10
#
# Populate it with upload_hf_data.py. Nothing here is gated, so notebooks 1
# and 4 still need no token at all.
DATA_REPO = "murugeshmarvel/caft-compliance-data"

# The exact set setup() insists on. If the dataset repo is missing any of
# these, it is better to say so plainly than to fail three cells later.
REQUIRED_FILES = [
    "alignment_data.json",
    "audit_data.json",
    "results/base_responses.json",
    "results/sft_responses.json",
    "results/caft_responses.json",
    "results/judge_scores.json",
    "results/utility_judge_summary.json",
]

REQUIREMENTS = [
    "transformers>=4.52",
    "peft>=0.11",
    "datasets",
    "pyarrow",
    "accelerate",
    "matplotlib",
    "pandas",
    "python-dotenv",
    "huggingface_hub",
]

_OK, _WARN, _BAD = "  ok ", " note", " FAIL"


def _line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<22} {detail}")


# ── 1. dependencies ───────────────────────────────────────────────────────────
def _clear_stale_torchao() -> str | None:
    """Remove Colab's preinstalled torchao if peft would reject it.

    peft calls is_torchao_available() from get_peft_model(). That function does
    not fail soft: if torchao is installed but older than peft's minimum, it
    raises ImportError and the LoRA cell dies. Colab currently ships torchao
    0.10.0 while recent peft wants >= 0.16.0, so every runtime hits this.

    Nothing here uses torchao — it is a quantisation library and we train in
    plain bfloat16 — so the safe fix is to remove it rather than chase a version
    that also has to match the runtime's torch. With the package gone,
    is_torchao_available() finds no module and returns False, which is the
    path peft is designed to take.
    """
    try:
        from importlib.metadata import version
        installed = version("torchao")
    except Exception:  # noqa: BLE001
        return None                      # not installed — nothing to do

    from packaging.version import parse
    if parse(installed) >= parse("0.16.0"):
        return None                      # new enough for peft, leave it alone

    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                   check=False)
    import importlib
    importlib.invalidate_caches()        # so find_spec stops seeing the old dir
    return installed


def install_requirements(in_colab: bool, quiet: bool = True) -> None:
    if not in_colab:
        _line(_OK, "dependencies", "local environment — nothing installed")
        return
    cmd = [sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS]
    subprocess.run(cmd, check=False)

    removed = _clear_stale_torchao()
    if removed:
        _line(_OK, "dependencies", f"installed for Colab (removed torchao {removed} "
                                   f"— too old for peft, and unused here)")
    else:
        _line(_OK, "dependencies", "installed for Colab")


# ── 2. project root and study data ────────────────────────────────────────────
def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for the folder that holds caft_colab.py."""
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "caft_colab.py").exists():
            return p
    return None


def fetch_study_data(
    repo_id: str = DATA_REPO,
    token: str | None = None,
    local_dir: Path | str | None = None,
) -> tuple[Path, str]:
    """Get the seven data files. Returns (folder, how-we-got-it).

    Local first: if `local_dir` already holds alignment_data.json, use it and
    touch the network not at all. That is the offline escape hatch — drop the
    files in `<repo>/data/` and the notebooks stop caring about the wifi.

    Otherwise pull the whole dataset repo in one `snapshot_download`. It is
    about 2 MB, public and ungated, and cached for the rest of the session.
    """
    if local_dir:
        local_dir = Path(local_dir)
        if (local_dir / "alignment_data.json").exists():
            return local_dir, f"local copy at {local_dir.name}/"

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "huggingface_hub is not installed, so the study data cannot be "
            "downloaded. Run:  pip install huggingface_hub"
        ) from e

    try:
        path = snapshot_download(repo_id=repo_id, repo_type="dataset", token=token)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download the study data from '{repo_id}'.\n"
            f"  {type(e).__name__}: {str(e)[:160]}\n\n"
            "  This dataset is public and ungated, so a failure here is almost\n"
            "  always one of three things:\n"
            "    - no internet on this runtime (Colab normally has it)\n"
            "    - the dataset repo does not exist yet — run upload_hf_data.py\n"
            "    - the repo is set to private on the Hub; make it public, or\n"
            "      pass a token that can read it\n\n"
            "  Fully offline alternative: put the seven files in <repo>/data/\n"
            "  and setup() will use those instead, without any network."
        ) from e
    return Path(path), f"Hub: {repo_id}"


# ── 3. Hugging Face token ─────────────────────────────────────────────────────
def find_hf_token(project_root: Path, in_colab: bool, prompt: bool = True) -> str | None:
    """Colab secret -> HF_TOKEN env var -> .env file -> interactive prompt."""
    if in_colab:
        try:
            from google.colab import userdata

            tok = userdata.get("HF_TOKEN")
            if tok:
                return tok.strip()
        except Exception:
            pass  # secret not set, or the user declined access — fall through

    tok = os.getenv("HF_TOKEN")
    if tok:
        return tok.strip()

    try:
        from dotenv import load_dotenv

        load_dotenv(project_root / ".env")
        tok = os.getenv("HF_TOKEN")
        if tok:
            return tok.strip()
    except Exception:
        pass

    if prompt and sys.stdin is not None:
        try:
            from getpass import getpass

            print(
                "\nNo Hugging Face token found.\n"
                "  1. Accept the licence at https://huggingface.co/google/medgemma-1.5-4b-it\n"
                "  2. Create a READ token at https://huggingface.co/settings/tokens\n"
                "  3. Paste it below (it is not echoed, and not saved to the notebook).\n"
                "  Press Enter to skip and use the ungated fallback model instead."
            )
            tok = getpass("HF token: ").strip()
            return tok or None
        except Exception:
            return None
    return None


# ── 4. gating check ───────────────────────────────────────────────────────────
def check_model_access(model_id: str, token: str | None) -> tuple[bool, str]:
    """Can we actually read this model's config? Returns (ok, reason)."""
    try:
        from huggingface_hub import model_info
    except Exception:
        return True, "huggingface_hub missing — skipping the check"

    try:
        model_info(model_id, token=token)
        return True, "licence accepted, token works"
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if "Gated" in name or "401" in str(e) or "403" in str(e):
            return False, "gated — licence not accepted, or token lacks access"
        if "RepositoryNotFound" in name or "404" in str(e):
            return False, "repo not found for this token"
        return False, f"{name}: {str(e)[:80]}"


# ── 5. device and dtype ───────────────────────────────────────────────────────
def _bf16_matmul_works(torch) -> bool:
    """Can this CUDA device actually multiply two bfloat16 matrices?

    bfloat16 *tensor cores* arrived with Ampere (compute capability 8.0), so
    the usual rule of thumb is "a T4 cannot do bf16". That rule is about speed,
    not about whether the maths runs at all — PyTorch will still execute a bf16
    matmul on a Turing card, just without tensor-core acceleration. Whether the
    installed cuBLAS accepts it varies by version, so stop guessing and try it.
    """
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        return bool(torch.isfinite(a @ a).all().item())
    except Exception:  # noqa: BLE001
        return False


def pick_device(strict: bool = True):
    """Return (device, dtype, human-readable name). Safe if torch is absent.

    With strict=False a GPU that cannot do bfloat16 is reported rather than
    refused, so the notebooks that only read data still run on it.

    Gemma-3 — and so MedGemma, which is built on it — cannot be trained in
    float16. Its residual stream grows layer by layer and, measured on this
    exact model with a clinical prompt, peaks at about 280,000. float16 stops
    at 65,504, so the residual becomes inf partway up the stack (layer 5 of
    34), the next RMSNorm turns that inf into nan, and every loss from step 1
    onwards is nan. This is a property of the model's activations rather than
    of any particular GPU, and it cannot be patched by keeping the adapters,
    the norm layers or the output head in float32 — the residual stream itself
    is the thing that overflows, and it is in the model's dtype.

    So: bfloat16 wherever it is available. bfloat16 has the same 16 bits as
    float16 but float32's exponent range, which covers 280,000 with room to
    spare.
    """
    try:
        import torch
    except Exception:
        return "cpu", None, "CPU only (torch not installed)"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if _bf16_matmul_works(torch):
            slow = torch.cuda.get_device_capability(0)[0] < 8
            note = ", bf16 has no tensor cores here — correct but slower" if slow else ""
            return "cuda", torch.bfloat16, f"{name}, {vram:.0f} GB{note}"
        # No usable bfloat16. float32 would be safe but a 4B model needs about
        # 17 GB of weights in float32 and will not fit. Say so plainly instead
        # of handing back float16 and producing nan losses in Demo 2.
        msg = (f"{name} cannot run bfloat16, and MedGemma produces nan losses "
               "in float16 (its activations reach about 280,000; float16 stops "
               "at 65,504). Use a runtime with an Ampere or newer GPU — on "
               "Colab, Runtime > Change runtime type > L4 or A100.")
        if strict:
            raise RuntimeError(msg)
        return "cuda", torch.float32, f"{name}, {vram:.0f} GB — NO BFLOAT16: {msg}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.bfloat16, "Apple Silicon (MPS)"
    return "cpu", torch.float32, "CPU only"


# ── the one function the notebooks call ───────────────────────────────────────
def setup(
    project_root: Path | str | None = None,
    need_gpu: bool = False,
    need_model: bool | None = None,
    install: bool = True,
    prompt_for_token: bool = True,
) -> SimpleNamespace:
    """Prepare the environment and report on it.

    need_gpu   : notebooks 2 and 3 train, so they want a GPU and a real model.
    need_model : defaults to `need_gpu`. Set False to skip the gating check
                 entirely (notebooks 1 and 4 never load a model).
    """
    need_model = need_gpu if need_model is None else need_model
    in_colab = "google.colab" in sys.modules

    print("=" * 74)
    print("CAFT notebook setup")
    print("=" * 74)

    install_requirements(in_colab, quiet=True) if install else None

    # paths
    root = Path(project_root).resolve() if project_root else find_project_root()
    if root is None:
        raise RuntimeError(
            "Could not find the project. Expected a folder containing "
            "'caft_colab.py'. On Colab, run the bootstrap cell that clones "
            "the repo before calling setup()."
        )
    _line(_OK, "project root", str(root))

    # study data — Hub, or a local copy in <repo>/data/ if one is there
    data_dir, source = fetch_study_data(DATA_REPO, token=None, local_dir=root / "data")
    _line(_OK, "study data", source)

    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"The study data at {data_dir} is incomplete. Missing:\n"
            + "".join(f"    {m}\n" for m in missing)
            + "\n  Re-run upload_hf_data.py to repopulate the dataset repo."
        )
    _line(_OK, "data files", f"all {len(REQUIRED_FILES)} present")

    align = data_dir / "alignment_data.json"
    audit = data_dir / "audit_data.json"
    results = data_dir / "results"

    # device
    device, dtype, gpu_name = pick_device(strict=bool(need_gpu))
    dt = str(dtype).replace("torch.", "") if dtype else "n/a"
    if need_gpu and device == "cpu":
        _line(_WARN, "device", "CPU — training will be unusably slow. "
                               "Colab: Runtime > Change runtime type > T4 GPU")
    elif "NO BFLOAT16" in gpu_name:
        _line(_WARN, "device", gpu_name)
    else:
        _line(_OK, "device", f"{device}, {dt} ({gpu_name})")

    # token + gating
    token, model_id, is_fallback = None, MEDGEMMA_ID, False
    if need_model:
        token = find_hf_token(root, in_colab, prompt=prompt_for_token)
        _line(_OK if token else _WARN, "hf token", "found" if token else "not found")

        ok, why = check_model_access(MEDGEMMA_ID, token)
        if ok:
            _line(_OK, "medgemma access", why)
        else:
            is_fallback = True
            model_id = FALLBACK_ID
            _line(_WARN, "medgemma access", why)
            print(
                "\n  Falling back to an ungated model so the notebook still runs:\n"
                f"      {FALLBACK_ID}\n"
                "  The code, the LoRA config and the CAFT loss are all identical.\n"
                "  The numbers will NOT match the talk — this model is small and\n"
                "  has no medical post-training. To use the real one:\n"
                "      1. accept the licence at\n"
                "         https://huggingface.co/google/medgemma-1.5-4b-it\n"
                "      2. create a READ token at\n"
                "         https://huggingface.co/settings/tokens\n"
                "      3. Colab: key icon in the left sidebar > add secret HF_TOKEN\n"
                "         local: put HF_TOKEN=... in a .env file next to the notebook\n"
                "      4. re-run this cell\n"
            )
        _line(_OK, "model id", model_id + ("  (FALLBACK)" if is_fallback else ""))
    else:
        _line(_OK, "model", "not needed — this notebook loads no weights")

    print("=" * 74)

    return SimpleNamespace(
        PROJECT_ROOT=root,
        DATA_DIR=data_dir,
        ALIGN_JSON=align,
        AUDIT_JSON=audit,
        RESULTS=results,
        MODEL_ID=model_id,
        HF_TOKEN=token,
        IS_FALLBACK=is_fallback,
        DEVICE=device,
        DTYPE=dtype,
        IN_COLAB=in_colab,
        GPU_NAME=gpu_name,
    )


# ── task dataset loader, shared by all four notebooks ─────────────────────────
def load_task_records(project_root: Path, as_dataframe: bool = False):
    """AlpaCare-MedInstruct-52k — 52,002 medical instruction/response pairs.

    Prefers a local Arrow copy at <root>/data/baseline_train if one is present,
    otherwise pulls the public (ungated) dataset from the Hub. The Hub path is
    the normal one on Colab: ~37 MB, a few seconds, cached for the session.
    """
    local = Path(project_root) / "data" / "baseline_train"
    if local.exists():
        from datasets import load_from_disk

        ds = load_from_disk(str(local))
        print(f"task data: local Arrow copy ({len(ds):,} rows)")
    else:
        from datasets import load_dataset

        try:
            ds = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not download the task dataset from Hugging Face.\n"
                f"  {type(e).__name__}: {str(e)[:160]}\n\n"
                "  This dataset is public and ungated, so a failure here is almost\n"
                "  always the network rather than permissions. Options:\n"
                "    - check the runtime has internet (Colab normally does)\n"
                "    - retry; the Hub is occasionally slow to respond\n"
                "    - or drop a local Arrow copy at <repo>/data/baseline_train\n"
                "      and this function will use it instead, entirely offline."
            ) from e
        print(f"task data: pulled from the Hub ({len(ds):,} rows)")
    return ds.to_pandas() if as_dataframe else list(ds)
