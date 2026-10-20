# Agent task: validate WIP TP-aware OOT RMSNorm patch on Azure

## Mission and source identity

You are testing a prototype, not preparing a success narrative. Find and fix
defects only within agreed scope, preserve red/green evidence, and do not open
a PR or request upstream review automatically.

Repository: https://github.com/shivankur-code/vllm
Branch: `fix/oot-tp-norm-overrides`
Initial patch commit: `a3d674c71b6cc371b94ae169a536fd4f2dbf7511`.
Upstream baseline: `34ac4f96a399836393894207dae296760b413715`.
Read root AGENTS.md, WIP-OOT-NORM-HANDOFF.md, and the repo PR checklist first.
This validation directory and the handoff document are WIP tooling, not files
to include in the eventual upstream PR.

The patch resolves a provider's registered norm replacement before adding a
cached TP-aware subclass. Exact-class-name dispatch previously missed base-norm
registrations. Relevant discussion: vllm-project/vllm#60218, especially
https://github.com/vllm-project/vllm/pull/60218#pullrequestreview-5429690332.
Recheck overlap and latest upstream before proposing a PR; do not blindly rebase
away the reproducible baseline while collecting evidence.

## Credentials, authorization and cost guardrails

- Never ask for Azure passwords, client secrets, tokens or private SSH keys in
  chat, prompt files, source, logs or environment dumps. Azure login is separate
  from GitHub access; the public branch can be cloned without GitHub credentials.
- The human signs in with `az login` in their own terminal/browser, or supplies
  an already configured managed identity/workload identity. If login requires a
  secret, human types it directly; do not capture it. Do not export access tokens.
- Obtain nonsecret subscription ID, allowed regions, maximum total spending,
  maximum runtime and approval to create/delete a dedicated resource group.
- Perform read-only discovery first. Verify credits cover this VM offer, quota,
  restrictions, price and likely availability. Quota/SKU listing is not capacity.
- Quote compute/hour + disk + network assumptions and setup/test time separately.
  Azure budget alerts are NOT hard spending caps. Do not promise a strict cap.
- No resource creation until the user approves the concrete SKU/region/cost and
  runtime. Avoid Spot initially: eviction makes debugging and evidence harder.
- Use only a dedicated, tagged resource group. No subscription-wide cleanup.
  Restrict SSH to the user's IP or use an approved Bastion/private route; never
  open SSH to the world. Do not open serving ports publicly.
- Set a runtime deadline, a backup scheduled shutdown/deallocation mechanism,
  and monitor it. Guest shutdown may not deallocate Azure compute. Disks/IPs can
  remain billable after deallocation. Export logs, then delete the dedicated
  group after explicit deletion approval; verify deletion completes.

## Machine requirements

Preferred: Ubuntu 22.04/24.04 x86_64; one VM with TWO full NVIDIA GPUs supported
by this vLLM/Torch version (prefer Ampere or newer, compute capability >=8.0).
An example to investigate is `Standard_NC48ads_A100_v4` (normally two A100s);
verify current SKU details, region, GPU count, offer eligibility and price.
It is an example, not a reservation or permission to provision it.

Practical host minimum: 16 vCPUs, 64 GiB RAM, 200 GiB available disk; prefer
32+ vCPUs, 128 GiB RAM and 256+ GiB SSD for dependency/build caches. Small models
fit well below A100 memory. A single GPU covers native/engine TP1; real TP2
requires two usable devices. TP4 is optional. Two single-GPU VMs introduce
multi-node networking/NCCL complexity and are not the first choice.

Use a supported Azure GPU image/driver installation procedure. No full CUDA
rebuild is intended: start from vLLM precompiled artifacts, pin/record versions.
If precompiled artifacts do not match this revision, stop and resolve the
compatibility/build plan rather than substituting a different vLLM wheel under
the source tree. Ensure `vllm.__file__` points to this checkout.

## Control-plane discovery (human-authenticated workstation)

Install Azure CLI if needed. Authenticate interactively, then run
`bash validation/oot_norm/azure_discover.sh SUBSCRIPTION_ID REGION OUTPUT_DIR`.
It performs read-only SKU/quota discovery and writes nonsecret account metadata.
Use Azure Retail Prices API/pricing calculator for a current regional Linux
on-demand rate, and the subscription billing/benefits view for credits. The
script does not verify credit eligibility, price or capacity and provisions
nothing. Keep account/quota outputs private, not in public Git commits.

## GPU VM bootstrap

Clone this branch onto Linux outside a shared checkout. Fetch the baseline if
the clone was shallow. Read every validation script before running it.

```bash
git clone --branch fix/oot-tp-norm-overrides https://github.com/shivankur-code/vllm.git
cd vllm
git rev-parse HEAD
nvidia-smi
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/lint.txt
uv pip install -r requirements/test/cuda.in
.venv/bin/pre-commit install
```

If `uv`/Git/compiler system prerequisites are absent, install per their official
instructions. Any privileged install is performed by the human/approved VM
bootstrap. Never use `sudo pip`, disable TLS, or log Hugging Face credentials.
The tiny-model fixture uses an ungated tokenizer but needs download access.

## Automated stages provided here

With a clean Git checkout and installed dependencies:

```bash
bash validation/oot_norm/run.sh native
bash validation/oot_norm/run.sh red
bash validation/oot_norm/run.sh tp
bash validation/oot_norm/run.sh engine-baseline
bash validation/oot_norm/run.sh lint
```

Each invocation creates its own UTC-stamped evidence directory under
`validation-results/oot-norm/` (Git-ignored), logs command output, records exit
status, and uses bounded command timeouts. An evidence directory is not a pass
certificate. Keep all phase directories and include failures in the final report.

Stages:

- `native`: new regression, full RMSNorm suite, all nearby fusers + registry tests.
  New test has 12 cases; collect JUnit. Simulated TP in that suite is not real TP.
- `red`: in this isolated clean checkout only, temporarily replaces ONLY fuser
  source with the pinned baseline while retaining added tests. Requires exactly
  12 call-phase AssertionError failures, no collection errors/skips, then restores
  original bytes even on ordinary failure/interrupt. Do not run concurrently with
  any other job. SIGKILL/VM loss can defeat Python cleanup; compare Git status and
  restore the fuser from the WIP commit if interrupted. Native green runs first.
- `tp`: runs real process-group validation with torchrun at TP1 and TP2, fresh
  processes for HW modes 0 and 1. Optional `RUN_TP4=1`. The real-TP script is newly
  authored and has NOT run on GPU locally: treat failures as evidence to diagnose.
  It asserts actual vendor execution plus independent numerical reference for
  both norm variants, dtype cases, noncontiguous inputs and uneven rejection.
- `engine-baseline`: existing tiny-Llama engine provider/logprob comparison.
  THIS DOES NOT test a loaded vendor override or compiled engine.
- `lint`: pre-commit on the two upstream patch files and whitespace check.

Scripts fail on the first failed stage command. Diagnose, preserve logs, fix and
rerun. Do not change assertions to make code pass without a justified contract.
Do not call the patch complete because these five commands finish successfully.

## Required remaining work (not automated by this package)

1. **Focused unit additions**: registration before/after fusion and removal,
   two same-name vendor classes, cached type reuse with independent instances,
   explicit supported invalid-override contract, no-plugin restoration. Extend
   the nearby suite. Keep tests concise and verify red/green for bug claims.
2. **Vendor engine integration**: extend the existing tiny-model e2e fixture
   with an importable startup plugin (vLLM general plugin entrypoint), loading in
   every spawn worker before model construction. Register supported provider
   bases; assert vendor instances/init/forward on EVERY worker. A numerically
   neutral sentinel implementation proves plumbing, not real vendor hardware.
3. **Normal and Gemma models**: use tiny local checkpoints for smoke tests; for
   model validation use an accessible pretrained model and the repo's relevant
   evaluation/logprob comparison. Fix prompts, seeds, dtype, checkpoint revision,
   decoding parameters and tolerance. Don't describe random weights as an
   accuracy evaluation. Avoid gated downloads unless human grants access safely.
4. **Eager + compiled, TP1 + TP2, HW0 + HW1**: no-plugin/reference vs vendor
   tokens/logprobs. Prove compilation is active, record compilation mode/backend,
   first compile and repeat/cache run. Assert vendor path is reached in compiled
   engine; avoid instrumentation that itself forces graph breaks. Use diagnostic
   counters separately if necessary. Cached dynamic class compatibility is a
   design risk, NOT an established compile failure or success.
5. **Serialization**: strict state-dict load is not whole-module pickling. Check
   real worker-spawn and compilation serialization in scope. If whole-module
   export is a supported requirement, test round-trip in a fresh process using
   importable vendor classes; fix wrapper design or explicitly agree scope.
6. **Actual vendor hardware**: only claim NVIDIA/sentinel validation unless the
   target accelerator and its real plugin have actually been exercised.

The existing mixin's sharded-residual path is unchanged; fuser matches single-input
norms. Do not claim general sharded fused-add support or expand scope silently.
No numerical performance/speedup claim; no benchmark is necessary for a pure
support/correctness claim, but serving correctness/model validation is required.

## Final verification and deliverables

Return a matrix of PASS / FAIL / NOT RUN / NOT APPLICABLE with evidence paths:
native red/green, nearby suites, TP1/2 (+ optional4), dtype/HW modes/norm variants,
no-plugin restoration, actual vendor eager engine, actual vendor compiled engine,
model eval, serialization, lint, human review, refreshed overlap check.

Include original and final commits, changed-file list, setup lock/freeze, Azure
SKU/GPU/driver/region, dependency compatibility, reproduction commands, exit codes,
JUnit totals including skips, representative numerical errors and known limits.
Store account info privately; redact credentials/prompts/IDs as appropriate before
public artifacts. Do not upload raw environment/credential/SSH files.

READY FOR HUMAN PR REVIEW only if all required gates have evidence and remaining
scope limits are accepted. Native collection failure is NOT a test failure that
proves the bug. No actual vendor execution means vendor integration is NOT proven.
Human must review every changed line/run relevant tests per AGENTS.md before any
PR (draft included); read and address `.agents/skills/pr-checklist/SKILL.md`.

Finally export all evidence BEFORE deletion, confirm no test processes still need
the VM, delete ONLY the approved dedicated resource group, verify resource deletion
and check for leftover disks/IPs/snapshots outside that group. Report actual billed
cost when available; do not infer it solely from estimated runtime. No PR creation
or review request without explicit authorization after the human gates.