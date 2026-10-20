"""Native CUDA/NCCL TP correctness probe; run via torch.distributed.run.

No model engine or compiler is tested. Newly authored, GPU execution outstanding.
"""
import importlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.transformers import layers
from vllm.model_executor.models.transformers.fusers import rms_norm


def main():
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    rows = []
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(size, rank, "env://", local, "nccl",
                                     timeout=timedelta(minutes=3))
        initialize_model_parallel(tensor_model_parallel_size=size, backend="nccl")
        try:
            dtypes = [torch.float32, torch.float16]
            if torch.cuda.is_bf16_supported():
                dtypes.append(torch.bfloat16)
            for zero_centered in (False, True):
                name = "GemmaRMSNorm" if zero_centered else "RMSNorm"
                base = layers._resolve("layernorm", name)
                assert getattr(rms_norm, name) is base
                package = "hw_agnostic." if ".hw_agnostic." in base.__module__ else ""
                registry = importlib.import_module(f"vllm.model_executor.{package}custom_op")
                saved = dict(registry.op_registry_oot)
                registry.op_registry_oot.pop(name, None)

                class SentinelNorm(base):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.initialized = True
                        self.calls = 0

                    def forward(self, x, residual=None):
                        self.calls += 1
                        return self.forward_native(x, residual)

                try:
                    base.register_oot(SentinelNorm)
                    for dtype in dtypes:
                        width = 32 * size
                        weight = torch.linspace(-0.5, 0.5, width, device=device)
                        source = torch.nn.RMSNorm(width, eps=1e-5, device=device,
                                                   dtype=dtype)
                        source.weight.data.copy_(weight)
                        fuser = rms_norm.RMSNormFuser(zero_centered, name, eps=1e-5)
                        built = fuser.fuse(source, "norm", None).to(device=device,
                                                                    dtype=dtype)
                        assert isinstance(built, SentinelNorm) and built.initialized
                        assert (built.tp_size, built.tp_rank) == (size, rank)
                        built.load_state_dict(source.state_dict(), strict=True)
                        assert list(built.state_dict()) == ["weight"]
                        x = torch.linspace(-2, 3, 4 * width, device=device,
                                           dtype=dtype).reshape(4, width)
                        scale = built.weight.float() + (1.0 if zero_centered else 0.0)
                        # Independent functional implementation, float accumulation.
                        expected = F.rms_norm(x.float(), (width,), scale, 1e-5).to(dtype)
                        tol = {torch.float32: 2e-5, torch.float16: 5e-3,
                               torch.bfloat16: 3e-2}[dtype]
                        with torch.no_grad():
                            full = built(x.clone())
                            torch.testing.assert_close(full, expected, rtol=tol, atol=tol)
                            shard = x.chunk(size, -1)[rank]
                            # Same values in an explicitly noncontiguous view.
                            storage = torch.empty((*shard.shape[:-1], shard.shape[-1] * 2),
                                                  device=device, dtype=dtype)
                            storage[..., ::2] = shard
                            noncontiguous = storage[..., ::2]
                            assert not noncontiguous.is_contiguous()
                            output = built(noncontiguous)
                            reference = expected.chunk(size, -1)[rank]
                            torch.testing.assert_close(output, reference, rtol=tol, atol=tol)
                            if size > 1:
                                try:
                                    built(x[..., :31])
                                except ValueError as exc:
                                    assert "does not tile it evenly" in str(exc)
                                else:
                                    raise AssertionError("Uneven shard was accepted")
                            assert built.calls == 2, "Uneven rejection must precede vendor call"
                        rows.append(dict(norm=name, provider=base.__module__,
                            dtype=str(dtype), tp=size, rank=rank, vendor_calls=built.calls,
                            max_abs_error=(output - reference).abs().max().item(),
                            outcome="pass"))
                finally:
                    registry.op_registry_oot.clear()
                    registry.op_registry_oot.update(saved)
            dist.barrier()
            out = Path(os.environ["OOT_EVIDENCE"])
            out.mkdir(parents=True, exist_ok=True)
            (out / f"rank-{rank}.json").write_text(json.dumps(dict(
                cases=rows, bf16_supported=torch.cuda.is_bf16_supported(),
                engine_tested=False, compilation_tested=False), indent=2))
            print(json.dumps({"rank": rank, "tp": size, "passed": len(rows)}), flush=True)
        finally:
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()