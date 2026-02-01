#!/usr/bin/env python3
"""
smoke_test.py (TAILORED) — dataset-free sanity checks for XCLIP_Baseline.

What it checks:
1) Repo structure files exist
2) Imports work: torch, transformers, xclip_config, xclip_module
3) Entrypoints respond to -h
4) Optional: a REAL forward pass using your get_model_and_processor (or XCLIPWrapper fallback)
   with dummy video frames + dummy text (NO dataset needed).

Usage:
  python smoke_test.py
  python smoke_test.py --verbose
  python smoke_test.py --run-forward
  python smoke_test.py --run-forward --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# -------------------------
# Pretty printing helpers
# -------------------------

def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def ok(msg: str) -> None:
    print(f"[OK]  {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")

# -------------------------
# Core checks
# -------------------------

def check_repo_files(repo_root: Path) -> bool:
    required = [
        "README.md",
        "xclip_config.py",
        "run_zeroshot.py",
        "run_linear_probe.py",
        "run_finetune.py",
        "xclip_module",
    ]
    all_ok = True
    for item in required:
        p = repo_root / item
        if p.exists():
            ok(f"exists: {item}")
        else:
            all_ok = False
            fail(f"missing: {item} ({p})")

    if (repo_root / "xclip_module").exists() and not (repo_root / "xclip_module").is_dir():
        all_ok = False
        fail("xclip_module exists but is not a directory")

    return all_ok

def check_environment(verbose: bool) -> bool:
    all_ok = True
    try:
        import torch  # type: ignore
        ok(f"torch import: torch=={torch.__version__}")

        cuda = torch.cuda.is_available()
        ok(f"cuda available: {cuda}")
        if cuda:
            try:
                name0 = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024 ** 3)
                ok(f"gpu[0]: {name0} | VRAM ~ {vram_gb:.1f} GB")
            except Exception as e:
                warn(f"could not read GPU details: {e}")
        else:
            warn("CUDA not available (fine for smoke test).")

    except Exception as e:
        all_ok = False
        fail(f"torch import failed: {e}")

    try:
        import transformers  # type: ignore
        ok(f"transformers import: transformers=={transformers.__version__}")
    except Exception as e:
        all_ok = False
        fail(f"transformers import failed: {e}")

    if verbose:
        ok(f"python: {sys.version.splitlines()[0]}")

    return all_ok

def check_imports(repo_root: Path) -> bool:
    all_ok = True
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import xclip_config  # type: ignore
        ok(f"import xclip_config: {getattr(xclip_config, '__file__', '')}")
    except Exception as e:
        all_ok = False
        fail(f"import xclip_config failed: {e}")

    try:
        import xclip_module  # type: ignore
        ok("import xclip_module")
    except Exception as e:
        all_ok = False
        fail(f"import xclip_module failed: {e}")

    return all_ok

def check_entrypoints_help(repo_root: Path, verbose: bool) -> bool:
    """
    Runs: python run_*.py -h
    """
    py = sys.executable
    scripts = ["run_zeroshot.py", "run_linear_probe.py", "run_finetune.py"]

    all_ok = True
    for s in scripts:
        p = repo_root / s
        if not p.exists():
            all_ok = False
            fail(f"{s} not found")
            continue

        cmd = [py, str(p), "-h"]
        if verbose:
            print("  $", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
                check=False,
            )
            if proc.returncode == 0:
                ok(f"{s} -h works")
            else:
                # Some scripts may not use argparse; warn rather than fail
                combined = (proc.stdout + "\n" + proc.stderr).strip().lower()
                if "unrecognized arguments" in combined or "unknown option" in combined:
                    warn(f"{s} -h not supported (consider argparse help)")
                else:
                    warn(f"{s} -h returned {proc.returncode} (might be doing work at import time)")
        except subprocess.TimeoutExpired:
            warn(f"{s} -h timed out (script might be doing heavy work at import time)")
        except Exception as e:
            warn(f"could not run {s} -h: {e}")

    return all_ok

# -------------------------
#  forward pass 
# -------------------------

def _build_dummy_video(num_frames: int, height: int = 224, width: int = 224) -> List[Any]:
    """
    Build dummy frames as numpy arrays (uint8 HxWxC).
    XCLIPProcessor accepts numpy frames, so we avoid PIL dependency.
    """
    import numpy as np  # type: ignore

    frames = []
    for _ in range(num_frames):
        frame = (np.random.rand(height, width, 3) * 255).astype("uint8")
        frames.append(frame)
    return frames

def load_model_and_processor(device_str: str) -> Tuple[Any, Any, str]:
    """
    Prefer your project helper: xclip_module.utils.get_model_and_processor
    Fallback: import XCLIPWrapper from model.py and use wrapper.processor
    """
    import torch  # type: ignore

    # Load config.MODEL_NAME if available
    model_name = None
    try:
        from xclip_config import config  # type: ignore
        model_name = getattr(config, "MODEL_NAME", None)
    except Exception:
        pass

    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")

    # Preferred path: the util function
    try:
        from xclip_module.utils import get_model_and_processor  # type: ignore
        if model_name is None:
            raise RuntimeError("config.MODEL_NAME is missing, cannot call get_model_and_processor safely.")
        model, processor = get_model_and_processor(model_name, device)
        return model, processor, f"Loaded via xclip_module.utils.get_model_and_processor('{model_name}')"
    except Exception as e:
        warn(f"get_model_and_processor path failed, falling back to XCLIPWrapper. Reason: {e}")

    # Fallback: wrapper
    try:
        from model import XCLIPWrapper  # type: ignore
        if model_name is None:
            model_name = "microsoft/xclip-base-patch32"  # safe default if config is missing
            warn(f"config.MODEL_NAME not found; using fallback model_name='{model_name}'")
        model = XCLIPWrapper(model_name=model_name)
        model = model.to(device)
        processor = model.processor
        return model, processor, f"Loaded via model.XCLIPWrapper('{model_name}')"
    except Exception as e:
        raise RuntimeError(f"Could not load model via fallback XCLIPWrapper either. Error: {e}")

def run_forward_pass(device_str: str, verbose: bool) -> bool:
    """
    Runs a real forward pass:
      inputs = processor(text=[...], videos=[[frames...]], return_tensors="pt", padding=True)
      logits = model(**inputs)
    Your wrapper returns logits_per_video (matrix).
    """
    import torch  # type: ignore

    # Load model+processor
    try:
        model, processor, how = load_model_and_processor(device_str=device_str)
        ok(how)
    except Exception as e:
        fail(str(e))
        return False

    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")

    # Build dummy batch (B=1 video, 1 text)
    # Note: XCLIPProcessor supports batch videos as list[list[frames]]
    num_frames = 8
    try:
        from xclip_config import config  # type: ignore
        num_frames = int(getattr(config, "EVAL_NUM_FRAMES", num_frames))
    except Exception:
        pass

    frames = _build_dummy_video(num_frames=num_frames, height=224, width=224)
    texts = ["dummy surgical query"]

    # Prepare inputs
    try:
        inputs = processor(
            text=texts,
            videos=[frames],          # batch size 1: a list containing one video (list of frames)
            return_tensors="pt",
            padding=True,
        )
    except Exception as e:
        fail(f"processor(...) failed: {e}")
        warn("Tip: this can fail if transformers/video dependencies are missing. Ensure 'transformers' is installed.")
        return False

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}

    # Forward
    try:
        model.eval()
        with torch.no_grad():
            # Using autocast only if CUDA
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(**inputs)
            else:
                logits = model(**inputs)

            # Handle DataParallel list output if that happens
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=0)

        # Print shape + diag
        if isinstance(logits, torch.Tensor):
            ok(f"forward pass OK | logits shape: {tuple(logits.shape)}")
            try:
                diag = logits.diag()
                ok(f"logits.diag() shape: {tuple(diag.shape)} | value sample: {diag[:1].cpu().tolist()}")
            except Exception:
                warn("Could not compute logits.diag() (shape may be unexpected).")

            if verbose:
                # Print keys used
                ok(f"input keys: {list(inputs.keys())}")

        else:
            warn(f"forward output is not a torch.Tensor (type={type(logits)}). Still OK, but unexpected.")

        return True

    except Exception as e:
        fail(f"model forward failed: {e}")
        warn("If this is the first run, model weights may need downloading (requires internet).")
        return False

# -------------------------
# Main
# -------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="XCLIP_Baseline smoke test (tailored).")
    parser.add_argument("--repo-root", type=str, default=".", help="Repo root path (default: .)")
    parser.add_argument("--verbose", action="store_true", help="More logs")
    parser.add_argument("--run-forward", action="store_true", help="Run a real model forward pass with dummy data")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu (default: cuda)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    header("SMOKE TEST: Repo files")
    files_ok = check_repo_files(repo_root)

    header("SMOKE TEST: Environment")
    env_ok = check_environment(verbose=args.verbose)

    header("SMOKE TEST: Imports")
    imports_ok = check_imports(repo_root)

    header("SMOKE TEST: Entrypoints (-h)")
    check_entrypoints_help(repo_root, verbose=args.verbose)

    forward_ok = True
    if args.run_forward:
        header("SMOKE TEST: Forward pass (dummy video + dummy text)")
        forward_ok = run_forward_pass(device_str=args.device, verbose=args.verbose)
    else:
        warn("Forward pass skipped. Run: python smoke_test.py --run-forward")

    header("SMOKE TEST: Summary")
    critical_ok = files_ok and env_ok and imports_ok and forward_ok
    if critical_ok:
        ok("All critical checks passed ✅")
        return 0
    else:
        fail("Some critical checks failed ❌ (see logs above)")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
