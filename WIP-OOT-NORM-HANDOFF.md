# WIP: OOT RMSNorm overrides through TP-aware fusers

**Branch-only handoff. Not a PR, reviewed fix, or validated accelerator implementation.**

Base: `34ac4f96a399836393894207dae296760b413715`.
Related limitation: https://github.com/vllm-project/vllm/pull/60218#pullrequestreview-5429690332.
AI assistance was used. Human review and native validation are still required.

## Current patch

The fuser resolves the current OOT replacement from the chosen provider's registry,
then creates a cached `(TPAwareNormMixin, VendorNorm)` subclass. No-plugin behavior
keeps the existing classes. The existing RMSNorm test suite has a new 12-case
regression for normal/Gemma norms, both provider selection modes and simulated
TP1/2/4. Gemma currently falls back to the in-tree provider in hw-agnostic mode.

Only these two implementation/test files belong in a future upstream PR:

- `vllm/model_executor/models/transformers/fusers/rms_norm.py`
- `tests/models/transformers/fusers/test_rms_norm.py`

Remove this handoff document from the eventual upstream PR diff. Do not submit
the current prototype until the gates below are satisfied.

## What has actually run

- Extracted-source harness with real CPU Torch 2.13.0: 12 expected baseline
  assertion failures, 12 patched passes. Platform dispatch and IR registration
  were stand-ins; collectives were simulated. **Not native vLLM testing.**
- Syntax and `git diff --check`: pass.
- Native pytest: collection failed on absent `tblib`; zero tests ran.
- Ruff/pre-commit: not run. Dependency downloads hit TLS handshake failures.
- Real TP collectives, compile/cache, model evaluation and accelerator: not run.

## Linux environment on the other machine

Use a supported Linux machine. A one-GPU CUDA machine suffices for native tests
and an initial model check; real TP2 needs two devices, TP4 needs four devices.
Neither a large model nor a four-GPU run is needed merely to reproduce dispatch.
Follow AGENTS.md and the current contributing guide if setup requirements change.

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/lint.txt
uv pip install -r requirements/test/cuda.in
.venv/bin/pre-commit install
```

Record the commit, OS, Python, Torch/vLLM versions, GPU model/driver and all test
outputs. Do not bypass TLS verification to install dependencies.

## Gate 1 — native red/green regression and surrounding suites

```bash
.venv/bin/python -m pytest tests/models/transformers/fusers/test_rms_norm.py::test_fuser_preserves_vendor_norm_and_tp -vv
.venv/bin/python -m pytest tests/models/transformers/fusers/test_rms_norm.py -vv
.venv/bin/python -m pytest tests/models/transformers/fusers/ tests/models/transformers/test_layer_registry.py -v
```

Acceptance:

- All 12 new cases pass in native vLLM with no source-extraction harness.
- Confirm the same new tests fail at vendor selection with the implementation
  reverted to the base; retain the added tests. Use an isolated worktree, not
  destructive changes to this branch. No red/green claim from collection errors.
- Existing detection, weightless, epsilon, native-norm numerics, no-plugin and
  uneven-sharding tests pass. Capture counts and skips; don't label skips passes.

**Important:** the added test simulates collectives even when run natively.
Passing it does not validate distributed process groups.

## Gate 2 — close missing behavioral coverage

Extend the existing suite with focused tests before claiming completion:

1. Registration lifetime: fuse before registration, register vendor, fuse again,
   then replace/remove the registration. The current registry must win each time;
   cached types must not freeze old selections.
2. Two independent vendor classes, same display name: no type-cache collisions.
3. Construct twice with one vendor: initialized instances are distinct, wrapper
   type is reused, weights are not shared.
4. Invalid/non-subclass override: agree and test the supported contract. Do not
   silently weaken existing resolver type checks or bypass vendor `__new__`.
5. No-plugin normal/Gemma outputs and state-dict keys before/after registration.
6. FP32 plus supported BF16/FP16, nontrivial weights and epsilon, full-width
   input with/without residual. Check both returned tensors on residual paths.
7. Weighted/weightless compatibility where supported; do not invent Gemma
   weightless support. Uneven shards must fail before any collective.

Sharded input plus residual is an existing unsupported/incorrectly handled
boundary in the current mixin; this patch does not repair or validate it. The
Transformers norm fuser matches single-input norms. Keep scope explicit and
do not claim general fused-add TP support without separate tests/design.

## Gate 3 — real TP2 correctness

No real-collective regression exists in this branch yet. Add one using the
repository's distributed helpers, or an explicitly documented `torchrun`
validation script. Do not present simulated tests as equivalent.

For both provider modes and both norm variants:

- Initialize real vLLM tensor-parallel process groups; obtain rank/size from them.
- Register a sentinel vendor implementation on every process before model/fuser
  construction; assert vendor init and forward actually execute on each rank.
- Create deterministic full-width input/weights, give each rank only its shard,
  call the actual fuser-built layer, and compare each rank's output to its slice
  of an independently normalized full tensor.
- Check TP1 and TP2, full-width inputs (must not gather), valid sharded inputs
  (must gather), noncontiguous shards, and uneven width rejection before gather.
- Verify checkpoint loading and run with supported FP32/BF16/FP16 tolerances.

TP2 is the minimum evidence of real distributed correctness for a TP fix. TP4 is
recommended if available; state explicitly if only TP2 was run. FP32 tolerances
can start from existing tests; choose dtype-appropriate tolerances with evidence.

## Gate 4 — engine and compilation

Existing runnable baseline engine test (local tiny random Llama checkpoint and
ungated tokenizer; real GPU job):

```bash
.venv/bin/python -m pytest tests/models/transformers/test_layer_registry_e2e.py::test_hw_agnostic_matches_vllm_end_to_end -vv
```

This only checks existing providers, **not the new vendor path**. Extend that
test infrastructure with a startup-loaded sentinel plugin and a tiny normal-norm
model, and use a tiny Gemma-style model for zero-centered coverage:

- Plugin loads before worker model imports, including spawn workers.
- Assert vendor norm instances and forward counters in every worker, not merely
  unchanged text output. Use a correctness-preserving vendor implementation.
- Fixed prompts and checkpoint; compare greedy tokens AND logprobs to an
  independent reference/no-plugin baseline using existing comparison helpers.
- Run TP1 and TP2, both provider modes, eager and the supported compiled engine
  mode. The existing test above explicitly uses eager and cannot prove compile.
- Record the actual compilation mode/backend; exercise first compile and cached
  repeat. Dynamic Python classes can affect guards/caches; source checks cannot
  establish compatibility.
- Run the repo-required model evaluation for the affected configuration and
  record accuracy/logprob comparison, configuration, seed and results. A random
  tiny-model smoke test is not a pretrained-model accuracy benchmark.

## Gate 5 — serialization and design decision

Strict `state_dict` weight loading is covered by the new unit test. Whole-module
pickling is NOT equivalent and is not established for cached dynamic classes.
Test the actual worker launch/compile serialization paths in scope. If whole
module round-tripping is required, add a round-trip with an importable vendor
class in a fresh process, and fix the design if it fails. Do not just publish
dynamic classes in `globals()` without handling name collisions/fresh imports.
Document any intentionally unsupported whole-module export contract.

## Gate 6 — hygiene and publication

```bash
.venv/bin/pre-commit run --files vllm/model_executor/models/transformers/fusers/rms_norm.py tests/models/transformers/fusers/test_rms_norm.py
git diff --check 34ac4f96a399836393894207dae296760b413715...HEAD
```

Before a PR (drafts included):

- Human reviews every changed line, understands the design and runs relevant
  tests per AGENTS.md. WIP commit is not an assertion of that review.
- Recheck upstream and competing PRs; coordinate with #60218 as needed.
- Rebase/update deliberately, rerun relevant tests, remove this handoff document.
- Run `.agents/skills/pr-checklist/SKILL.md`, include AI disclosure, exact native
  commands/results, model validation, skips and limitations in the PR body.
- Add any required DCO sign-off only after the human can make that certification.

No performance benchmark is needed for a correctness/support-only claim. Do not
claim a real vendor accelerator works until that backend is actually exercised.