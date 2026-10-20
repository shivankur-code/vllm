"""Bounded Linux validation stages; successful stages do NOT certify PR readiness."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASE = "34ac4f96a399836393894207dae296760b413715"
FUSER = "vllm/model_executor/models/transformers/fusers/rms_norm.py"
TEST = "tests/models/transformers/fusers/test_rms_norm.py"
NODE = TEST + "::test_fuser_preserves_vendor_norm_and_tp"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["native", "red", "tp", "engine-baseline", "lint"])
    args = parser.parse_args()
    if sys.platform != "linux":
        raise SystemExit("Use a supported Linux GPU environment; no CPU stand-ins here.")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("Commit/review changes first: validation requires a clean checkout.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "validation-results" / "oot-norm" / f"{stamp}-{args.phase}"
    out.mkdir(parents=True)
    report = dict(phase=args.phase, commit=git("rev-parse", "HEAD"), baseline=BASE,
                  python=sys.version, commands=[], status="running", pr_ready=False,
                  missing_gates=["registration-lifetime tests", "vendor engine eager/compiled",
                                 "model evaluation", "serialization scope", "human review"])

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2))

    def run(label, command, seconds=1800, extra_env=None, expected=0):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        if extra_env:
            env.update(extra_env)
        log = out / f"{label}.log"
        row = dict(label=label, command=command, timeout_seconds=seconds,
                   expected_exit=expected, log=log.name)
        report["commands"].append(row)
        save()
        print(f"Running {label}; log: {log}", flush=True)
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=seconds)
            except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                row["aborted"] = True
                save()
                raise
        row["exit_code"] = code
        save()
        if code != expected:
            raise RuntimeError(f"{label}: exit {code}, expected {expected}; inspect {log}")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        save()
        run("import-check", [sys.executable, "-c",
            "import pathlib,torch,vllm; print('vllm',vllm.__file__); "
            f"assert pathlib.Path(vllm.__file__).resolve().is_relative_to(pathlib.Path({str(ROOT)!r})); "
            "print('torch',torch.__version__,'cuda',torch.version.cuda); "
            "assert torch.cuda.is_available(); print('gpu_count',torch.cuda.device_count()); "
            "print([(torch.cuda.get_device_name(i),torch.cuda.get_device_capability(i)) "
            "for i in range(torch.cuda.device_count())])"], seconds=120)
        run("gpu-driver", ["nvidia-smi"], seconds=60)
        report["packages"] = {p: importlib.metadata.version(p) for p in
                              ("torch", "pytest", "transformers")}
        if args.phase == "native":
            for label, targets in [("new-regression", [NODE]), ("rms-suite", [TEST]),
                ("nearby", ["tests/models/transformers/fusers/",
                            "tests/models/transformers/test_layer_registry.py"])]:
                run(label, [sys.executable, "-m", "pytest", *targets, "-vv",
                            f"--junitxml={out / (label + '.xml')}"])
            xml = ET.parse(out / "new-regression.xml")
            cases = list(xml.iter("testcase"))
            assert len(cases) == 12 and all(
                not any(c.tag in {"failure", "error", "skipped"} for c in case)
                for case in cases), "Expected 12 actual passing cases, no skips"
        elif args.phase == "red":
            path = ROOT / FUSER
            original = path.read_bytes()
            (out / "patched-fuser-backup.py").write_bytes(original)
            baseline = subprocess.check_output(["git", "show", f"{BASE}:{FUSER}"], cwd=ROOT)
            try:
                path.write_bytes(baseline)
                run("baseline-regression", [sys.executable, "-m", "pytest", NODE,
                    "-vv", f"--junitxml={out / 'red.xml'}"], expected=1)
                cases = list(ET.parse(out / "red.xml").iter("testcase"))
                assert len(cases) == 12, "Red result must contain all 12 cases"
                for case in cases:
                    failure = case.find("failure")
                    assert failure is not None and "AssertionError" in (
                        failure.get("message", "") + (failure.text or ""))
                    assert "assert isinstance(built, VendorNorm)" in (failure.text or ""), (
                        "Baseline must fail at vendor selection, not an unrelated assertion")
                    assert case.find("error") is None and case.find("skipped") is None
            finally:
                path.write_bytes(original)
                assert path.read_bytes() == original
            assert not git("diff", "--", FUSER), "Baseline stage did not restore fuser"
        elif args.phase == "tp":
            import torch
            if torch.cuda.device_count() < 2:
                raise RuntimeError("TP phase requires >=2 GPUs; TP1 alone is insufficient")
            sizes = (1, 2, 4) if os.environ.get("RUN_TP4") == "1" else (1, 2)
            for size in sizes:
                assert torch.cuda.device_count() >= size
                for hw in ("0", "1"):
                    evidence = out / f"tp{size}-hw{hw}"
                    run(f"tp{size}-hw{hw}", [sys.executable, "-m", "torch.distributed.run",
                        "--standalone", f"--nproc-per-node={size}",
                        "validation/oot_norm/real_tp.py"], seconds=900,
                        extra_env={"VLLM_USE_HW_AGNOSTIC": hw,
                                   "OOT_EVIDENCE": str(evidence),
                                   "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"})
                    for rank in range(size):
                        result = json.loads((evidence / f"rank-{rank}.json").read_text())
                        assert len(result["cases"]) >= 4
                        assert all(r["outcome"] == "pass" for r in result["cases"])
        elif args.phase == "engine-baseline":
            run("engine-baseline", [sys.executable, "-m", "pytest",
                "tests/models/transformers/test_layer_registry_e2e.py::"
                "test_hw_agnostic_matches_vllm_end_to_end", "-vv",
                f"--junitxml={out / 'engine.xml'}"], seconds=3600)
            cases = list(ET.parse(out / "engine.xml").iter("testcase"))
            assert cases and all(not any(c.tag in {"failure", "error", "skipped"}
                                         for c in case) for case in cases)
            report["scope"] = "No-plugin eager engine only; vendor/compiled engine NOT tested"
        else:
            run("precommit", [str(ROOT / ".venv/bin/pre-commit"), "run", "--files", FUSER, TEST])
            run("whitespace", ["git", "diff", "--check", f"{BASE}...HEAD"])
        report["status"] = "stage_passed_not_pr_ready"
    except BaseException as exc:
        report.update(status="failed_or_aborted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["final_git_status"] = git("status", "--porcelain", "--untracked-files=normal")
        save()
        print(f"Evidence: {out}; PR readiness is NOT certified.", flush=True)


if __name__ == "__main__":
    main()