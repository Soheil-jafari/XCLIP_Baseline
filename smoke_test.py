#!/usr/bin/env python3
"""
smoke_test.py — dataset-free sanity checks for the XCLIP_Baseline repo.

Usage:
  python smoke_test.py
  python smoke_test.py --verbose
  python smoke_test.py --no-subprocess

Optional (if you know a model import path and want a real forward pass):
  python smoke_test.py --forward "xclip_module.some_module:XCLIP" --device cuda --fp16
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

# ---------------------------
# Helpers
# ---------------------------

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def _ok(name: str, detail: str = "") -> CheckResult:
    return CheckResult(name=name, ok=True, detail=detail)

def _fail(name: str, detail: str = "") -> CheckResult:
    return CheckResult(name=name, ok=False, detail=detail)

def _warn(name: str, detail: str = "") -> CheckResult:
    # Warnings are treated as "ok" but we mark them clearly in output.
    return CheckResult(name=name, ok=True, detail=f"[WARN] {detail}")

def _run_subprocess(cmd: list[str], timeout_s: int, verbose: bool) -> Tuple[int, str, str]:
    if verbose:
        print(f"  $ {' '.join(cmd)}")
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return p.returncode, (p.stdout or ""), (p.stderr or "")

def _import_by_path(path: str) -> Any:
    """
    Import helper for strings like:
      "xclip_module.some_module:ClassOrFuncName"
    """
    if ":" not in path:
        raise ValueError('Expected format "module.submodule:Name"')
    mod_path, name = path.split(":", 1)
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, name):
        raise AttributeError(f'Module "{mod_path}" has no attribute "{name}"')
    return getattr(mod, name)

def _safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(signature unavailable)"

# ---------------------------
# Checks
# ---------------------------

def check_repo_files(repo_root: Path) -> list[CheckResult]:
    required = [
        "README.md",
        "requirements.txt",
        "xclip_config.py",
        "run_zeroshot.py",
        "run_linear_probe.py",
        "run_finetune.py",
        "xclip_module",
    ]
    results: list[CheckResult] = []

    for item in required:
        p = repo_root / item
        if p.exists():
            results.append(_ok(f"exists: {item}"))
        else:
            results.append(_fail(f"exists: {item}", f"Missing: {p}"))

    # sanity: xclip_module should be a directory
    p = repo_root / "xclip_module"
    if p.exists() and not p.is_dir():
        results.append(_fail("xclip_module is a directory", f"Found a file at {p}"))
    elif p.exists():
        results.append(_ok("xclip_module is a directory"))

    return results

def check_environment(verbose: bool) -> list[CheckResult]:
    results: list[CheckResult] = []

    results.append(_ok("python", f"{sys.version.splitlines()[0]}"))
    results.append(_ok("platform", f"{platform.platform()}"))

    # Torch + CUDA
    try:
        import torch  # type: ignore

        results.append(_ok("torch import", f"torch=={torch.__version__}"))
        results.append(_ok("cuda available", str(torch.cuda.is_available())))

        if torch.cuda.is_available():
            try:
                dev_count = torch.cuda.device_count()
                results.append(_ok("cuda device_count", str(dev_count)))
                if dev_count > 0:
                    name0 = torch.cuda.get_device_name(0)
                    props = torch.cuda.get_device_properties(0)
                    vram_gb = props.total_memory / (1024 ** 3)
                    results.append(_ok("cuda device[0]", f"{name0} | VRAM ~ {vram_gb:.1f} GB"))
            except Exception as e:
                results.append(_warn("cuda device info", f"Could not query device details: {e}"))
        else:
            results.append(_warn("cuda", "CUDA not available. That is OK for a smoke test."))

    except Exception as e:
        results.append(_fail("torch import", f"Failed to import torch: {e}"))

    return results

def check_imports(repo_root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    # ensure repo root is on sys.path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # import xclip_config
    try:
        import xclip_config  # type: ignore

        results.append(_ok("import xclip_config", f"Loaded from {getattr(xclip_config, '__file__', 'unknown')}"))
    except Exception as e:
        results.append(_fail("import xclip_config", f"{e}"))

    # import xclip_module
    try:
        import xclip_module  # type: ignore

        results.append(_ok("import xclip_module", f"Loaded from {getattr(xclip_module, '__file__', 'package')}"))
    except Exception as e:
        results.append(_fail("import xclip_module", f"{e}"))

    return results

def check_entrypoints_help(repo_root: Path, verbose: bool, timeout_s: int = 20) -> list[CheckResult]:
    """
    Runs:
      python run_*.py -h
    If scripts don't use argparse and error on -h, we downgrade to warning (not failure).
    """
    results: list[CheckResult] = []
    py = sys.executable

    scripts = ["run_zeroshot.py", "run_linear_probe.py", "run_finetune.py"]
    for s in scripts:
        p = repo_root / s
        if not p.exists():
            results.append(_fail(f"{s} help", "Script not found"))
            continue

        cmd = [py, str(p), "-h"]
        try:
            rc, out, err = _run_subprocess(cmd, timeout_s=timeout_s, verbose=verbose)
            combined = (out + "\n" + err).strip()

            if rc == 0:
                # likely argparse help printed
                results.append(_ok(f"{s} -h", "OK"))
            else:
                # If it looks like "unrecognized arguments: -h" or similar, treat as warning.
                lowered = combined.lower()
                if "unrecognized arguments" in lowered or "unknown option" in lowered or "no such option" in lowered:
                    results.append(_warn(f"{s} -h", "Script may not use argparse; consider adding CLI help."))
                else:
                    results.append(_warn(f"{s} -h", f"Non-zero exit ({rc}). Output:\n{combined[:400]}"))
        except subprocess.TimeoutExpired:
            results.append(_warn(f"{s} -h", f"Timed out after {timeout_s}s (script might be doing work at import time)."))
        except Exception as e:
            results.append(_warn(f"{s} -h", f"Could not run help check: {e}"))

    return results

def best_effort_find_model(verbose: bool = False) -> Tuple[Optional[str], Optional[Any], str]:
    """
    Tries to discover a likely XCLIP class or a build/create function inside xclip_module.
    Returns: (kind, obj, message)
      kind: "class" | "factory" | None
    """
    try:
        import xclip_module  # type: ignore
    except Exception as e:
        return None, None, f"Cannot import xclip_module: {e}"

    import pkgutil

    candidates: list[str] = []
    try:
        for m in pkgutil.walk_packages(xclip_module.__path__, prefix=xclip_module.__name__ + "."):
            modname = m.name
            low = modname.lower()
            if any(k in low for k in ["xclip", "model", "clip"]):
                candidates.append(modname)
    except Exception as e:
        return None, None, f"Could not enumerate submodules: {e}"

    # limit to avoid long imports
    candidates = candidates[:25]

    best_msg_parts = []
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            if verbose:
                best_msg_parts.append(f"- skip {modname}: import error: {e}")
            continue

        # Search for factory functions first
        for name, obj in vars(mod).items():
            if callable(obj) and name.lower() in {"build_model", "create_model", "get_model", "load_model", "build_xclip", "create_xclip"}:
                return "factory", obj, f"Found factory: {modname}:{name} { _safe_signature(obj) }"

        # Then search for likely model classes
        for name, obj in vars(mod).items():
            if isinstance(obj, type) and "xclip" in name.lower():
                return "class", obj, f"Found class: {modname}:{name} { _safe_signature(obj) }"

    msg = "No obvious XCLIP class/factory auto-discovered."
    if verbose and best_msg_parts:
        msg += "\nDetails:\n" + "\n".join(best_msg_parts[:8])
    return None, None, msg

def optional_forward_pass(model_path: str, device: str, fp16: bool, verbose: bool) -> CheckResult:
    """
    Attempts a tiny forward pass with dummy inputs for a user-specified model class/function.
    This is OPTIONAL because each repo implementation differs.
    """
    try:
        import torch  # type: ignore
    except Exception as e:
        return _fail("forward pass", f"torch import failed: {e}")

    try:
        obj = _import_by_path(model_path)
    except Exception as e:
        return _fail("forward pass", f"Could not import '{model_path}': {e}")

    # Instantiate (best-effort)
    model = None
    try:
        if isinstance(obj, type):
            # try no-arg init
            model = obj()
        elif callable(obj):
            # try no-arg factory
            model = obj()
        else:
            return _fail("forward pass", f"Target is not callable/class: {type(obj)}")
    except Exception as e:
        return _fail(
            "forward pass",
            f"Could not instantiate model from '{model_path}'. "
            f"Try pointing to a different builder/class.\nError: {e}"
        )

    if not hasattr(model, "forward"):
        return _fail("forward pass", "Instantiated object has no forward() method.")

    # Device
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev)
    model.eval()

    # Dummy inputs — common video tensor layout: (B, T, C, H, W)
    B, T, C, H, W = 1, 4, 3, 64, 64
    video = torch.randn(B, T, C, H, W, device=dev)

    # Dummy text tokens — we don't assume a tokenizer exists
    # Many transformer text encoders accept (B, L) int tokens
    text_tokens = torch.randint(low=0, high=1000, size=(B, 16), device=dev)

    # Forward
    try:
        with torch.no_grad():
            if fp16 and dev.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(video, text_tokens)  # type: ignore
            else:
                out = model(video, text_tokens)  # type: ignore

        # Summarize output type/shape
        if verbose:
            print("Forward output type:", type(out))
            try:
                import torch  # type: ignore
                if isinstance(out, torch.Tensor):
                    print("Forward output shape:", tuple(out.shape))
                elif isinstance(out, (tuple, list)):
                    shapes = []
                    for x in out:
                        if hasattr(x, "shape"):
                            shapes.append(tuple(x.shape))  # type: ignore
                        else:
                            shapes.append(type(x).__name__)
                    print("Forward output (tuple/list) shapes:", shapes)
                elif isinstance(out, dict):
                    keys = list(out.keys())
                    print("Forward output dict keys:", keys[:20])
            except Exception:
                pass

        return _ok("forward pass", f"OK ({model_path})")

    except TypeError as e:
        return _fail(
            "forward pass",
            "Model forward signature didn't match (common in different implementations).\n"
            f"Try another --forward path or adjust dummy inputs.\nError: {e}"
        )
    except Exception as e:
        return _fail("forward pass", f"Forward failed: {e}")

# ---------------------------
# Main
# ---------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset-free smoke test for XCLIP_Baseline.")
    parser.add_argument("--repo-root", type=str, default=".", help="Path to repo root (default: .)")
    parser.add_argument("--verbose", action="store_true", help="Print more debug output")
    parser.add_argument("--no-subprocess", action="store_true", help="Skip running entry scripts with -h")
    parser.add_argument("--forward", type=str, default="", help='Optional forward test: "module.sub:ClassOrFunc"')
    parser.add_argument("--device", type=str, default="cuda", help="Device for optional forward pass (cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use autocast FP16 for optional forward pass (CUDA only)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    _print_header("SMOKE TEST: Repo files")
    file_results = check_repo_files(repo_root)
    for r in file_results:
        print(f"[{'OK' if r.ok else 'FAIL'}] {r.name}" + (f" — {r.detail}" if r.detail else ""))

    _print_header("SMOKE TEST: Environment")
    env_results = check_environment(verbose=args.verbose)
    for r in env_results:
        print(f"[{'OK' if r.ok else 'FAIL'}] {r.name}" + (f" — {r.detail}" if r.detail else ""))

    _print_header("SMOKE TEST: Imports")
    import_results = check_imports(repo_root)
    for r in import_results:
        print(f"[{'OK' if r.ok else 'FAIL'}] {r.name}" + (f" — {r.detail}" if r.detail else ""))

    if not args.no_subprocess:
        _print_header("SMOKE TEST: Entrypoints (-h)")
        entry_results = check_entrypoints_help(repo_root, verbose=args.verbose)
        for r in entry_results:
            print(f"[{'OK' if r.ok else 'FAIL'}] {r.name}" + (f" — {r.detail}" if r.detail else ""))

    _print_header("SMOKE TEST: Model auto-discovery (best effort)")
    kind, obj, msg = best_effort_find_model(verbose=args.verbose)
    if kind is None:
        print(f"[OK] auto-discovery — {msg}")
    else:
        print(f"[OK] auto-discovery — {msg}")

    if args.forward.strip():
        _print_header("SMOKE TEST: Optional forward pass")
        r = optional_forward_pass(args.forward.strip(), device=args.device, fp16=args.fp16, verbose=args.verbose)
        print(f"[{'OK' if r.ok else 'FAIL'}] {r.name}" + (f" — {r.detail}" if r.detail else ""))

    # Final verdict: fail only if critical checks failed
    critical = file_results + env_results + import_results
    failed = [r for r in critical if not r.ok]

    _print_header("SMOKE TEST: Summary")
    if failed:
        print("FAILED critical checks:")
        for r in failed:
            print(f" - {r.name}: {r.detail}")
        return 1

    print("All critical checks passed ✅")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
