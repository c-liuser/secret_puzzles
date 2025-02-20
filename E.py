### Apache 2.0 License Header ###
"""
Copyright (C) 2025 Cole Liu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


############################################
# Helper: Compute Maximum Chunk Size with Safety Factor
############################################
def compute_chunk_size(
    vocab: int,
    available_vram_gb: float,
    bytes_per_element: int = 4,
    safety_factor: float = 28,
) -> int:
    """
    Computes the maximum chunk size (number of tokens processed per chunk) based on the theoretical
    VRAM usage and applies a safety factor to account for overhead.
    """
    available_bytes = available_vram_gb * (1024**3)
    max_chunk = available_bytes // (vocab * bytes_per_element)
    safe_chunk = max_chunk // safety_factor
    return max(1, int(safe_chunk))


############################################
# Helper: Print Peak VRAM Usage
############################################
def print_max_vram_usage(msg: str = ""):
    max_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"{msg}Max VRAM usage: {max_mem:.2f} MB")


############################################
# 1. Convenience transformation function
############################################
def transformation_function(batch, linear, labels):
    """
    Applies the linear projection to the input batch and then computes
    cross-entropy loss. The projection is done in half precision,
    then upcast to float32 for numerical stability.

    batch:  [chunk_size, D] (half precision)
    linear: nn.Linear(D, vocab) (half precision)
    labels: [chunk_size]
    """
    x = linear(batch).float()  # up-projection, cast to float32
    ce_loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss = ce_loss_fn(x.view(-1, x.shape[-1]), labels.view(-1))
    return loss


###########################################################
# 2. Memory-efficient Cross-Entropy autograd function (chunked computation)
###########################################################
class MemoryEfficientCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_feats: torch.Tensor,
        linear: nn.Linear,
        targets: torch.Tensor,
        chunk_size: int,
    ):
        device = input_feats.device

        # If input is [B, Q, D], flatten to [N, D]
        if input_feats.dim() == 3:
            B, Q, D = input_feats.shape
            N = B * Q
            x_flat = input_feats.view(N, D)
            t_flat = targets.view(N)
            is_3d = True
        else:
            N, D = input_feats.shape
            x_flat = input_feats
            t_flat = targets
            B, Q = None, None
            is_3d = False

        # Accumulate the sum of losses; later average over total tokens.
        sum_loss = torch.zeros([], device=device, dtype=torch.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_flat[start:end]
            t_chunk = t_flat[start:end]
            chunk_loss = transformation_function(x_chunk, linear, t_chunk)
            current_size = end - start
            sum_loss += chunk_loss * current_size

        total_loss = sum_loss / float(N)

        ctx.save_for_backward(input_feats, targets)
        ctx.linear = linear
        ctx.chunk_size = chunk_size
        ctx.is_3d = is_3d
        ctx.bq = (B, Q)
        ctx.N, ctx.D = N, D

        return total_loss

    @staticmethod
    def backward(ctx, grad_output):
        input_feats, targets = ctx.saved_tensors
        linear = ctx.linear
        chunk_size = ctx.chunk_size
        is_3d = ctx.is_3d
        B, Q = ctx.bq
        N, D = ctx.N, ctx.D

        W = linear.weight  # [vocab, D]
        b = linear.bias  # [vocab] or None

        if is_3d:
            x_flat = input_feats.view(N, D)
            t_flat = targets.view(N)
        else:
            x_flat = input_feats
            t_flat = targets

        grad_in = torch.zeros_like(x_flat)  # dL/dX: [N, D]
        grad_w = torch.zeros_like(W)  # dL/dW: [vocab, D]
        grad_b = torch.zeros_like(b) if b is not None else None

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_flat[start:end].float()  # cast to float32 for stability
            t_chunk = t_flat[start:end]
            M = end - start

            logits_chunk = x_chunk.matmul(W.t().float())
            if b is not None:
                logits_chunk += b
            max_vals = logits_chunk.max(dim=1, keepdim=True)[0]
            logits_chunk = logits_chunk - max_vals
            exp_chunk = logits_chunk.exp()
            sum_exp = exp_chunk.sum(dim=1, keepdim=True)
            probs = exp_chunk / sum_exp

            idx = torch.arange(M, device=probs.device)
            probs[idx, t_chunk] -= 1.0
            probs /= float(N)

            grad_in_chunk = probs.matmul(W.float())
            grad_in[start:end] = grad_in_chunk.to(grad_in.dtype)
            grad_w_chunk = probs.t().matmul(x_chunk)
            grad_w.add_(grad_w_chunk.to(grad_w.dtype))
            if grad_b is not None:
                grad_b_chunk = probs.sum(dim=0)
                grad_b.add_(grad_b_chunk.to(grad_b.dtype))

        grad_in *= grad_output
        grad_w *= grad_output
        if grad_b is not None:
            grad_b *= grad_output

        if is_3d:
            grad_in = grad_in.view(B, Q, D)

        if linear.weight.grad is None:
            linear.weight.grad = torch.zeros_like(linear.weight)
        linear.weight.grad.add_(grad_w)
        if linear.bias is not None:
            if linear.bias.grad is None:
                linear.bias.grad = torch.zeros_like(linear.bias)
            linear.bias.grad.add_(grad_b)

        return grad_in, None, None, None


###########################################################
# 3. Convenience wrapper function
###########################################################
def memory_efficient_cross_entropy_with_transform(
    input_feats: torch.Tensor,
    linear_module: nn.Linear,
    targets: torch.Tensor,
    chunk_size: int = 8192,
):
    return MemoryEfficientCEFunction.apply(
        input_feats, linear_module, targets, chunk_size
    )


###########################################################
# 4. Test 1: Gradient Equivalence Test
###########################################################
def test_gradient_equivalence():
    """Test that the memory-efficient CE produces equivalent gradients to the normal full-logits version."""
    print("Testing gradient equivalence on moderate-sized inputs...")
    bsz, qlen, hd, vocab = 2, 256, 64, 1024

    X = torch.randn(
        bsz, qlen, hd, device="cuda", dtype=torch.float16, requires_grad=True
    )
    T = torch.randint(0, vocab, (bsz, qlen), device="cuda")
    linear = nn.Linear(hd, vocab, bias=True).cuda().half()

    # Clone inputs and module for both approaches.
    X_normal = X.clone().detach().requires_grad_(True)
    X_mem = X.clone().detach().requires_grad_(True)
    linear_normal = nn.Linear(hd, vocab, bias=True).cuda().half()
    linear_mem = nn.Linear(hd, vocab, bias=True).cuda().half()
    linear_normal.load_state_dict(linear.state_dict())
    linear_mem.load_state_dict(linear.state_dict())

    def normal_ce_loss(input_feats, linear, targets):
        B, Q, D = input_feats.shape
        logits = linear(input_feats.view(-1, D)).float()  # full logits in float32
        loss = F.cross_entropy(logits, targets.view(-1))
        return loss

    torch.cuda.reset_peak_memory_stats()
    loss_normal = normal_ce_loss(X_normal, linear_normal, T)
    loss_normal.backward()
    grad_X_normal = X_normal.grad.clone()
    grad_W_normal = linear_normal.weight.grad.clone()
    grad_b_normal = (
        linear_normal.bias.grad.clone() if linear_normal.bias is not None else None
    )

    print(f"Normal approach loss: {loss_normal.item():.6f}")
    print_max_vram_usage("Normal approach: ")

    torch.cuda.reset_peak_memory_stats()
    # Use a fixed chunk size for the moderate test.
    loss_mem = memory_efficient_cross_entropy_with_transform(
        X_mem, linear_mem, T, chunk_size=128
    )
    loss_mem.backward()
    grad_X_mem = X_mem.grad.clone()
    grad_W_mem = linear_mem.weight.grad.clone()
    grad_b_mem = linear_mem.bias.grad.clone() if linear_mem.bias is not None else None

    print(f"Memory-efficient approach loss: {loss_mem.item():.6f}")
    print_max_vram_usage("Memory-efficient approach: ")

    print(
        "Loss close:",
        torch.allclose(
            torch.tensor(loss_normal.item()), torch.tensor(loss_mem.item()), atol=1e-4
        ),
    )
    print(
        "Gradient X close:",
        torch.allclose(grad_X_normal.float(), grad_X_mem.float(), atol=1e-4),
    )
    print(
        "Gradient W close:",
        torch.allclose(grad_W_normal.float(), grad_W_mem.float(), atol=1e-4),
    )
    if grad_b_normal is not None and grad_b_mem is not None:
        print(
            "Gradient b close:",
            torch.allclose(grad_b_normal.float(), grad_b_mem.float(), atol=1e-4),
        )


###########################################################
# 5. Test 2: Large Input Test
###########################################################
def test_large_input():
    """Test that the memory-efficient CE does not OOM on large inputs.
    We use a computed safe chunk size based on theoretical VRAM usage (with a safety factor).
    """
    print("\nTesting large input (memory-efficient) for OOM...")
    bsz = 4
    qlen = 4096
    hd = 4096
    vocab = 128 * 1024  # 131072
    available_vram_gb = 14.0  # Allow wiggle room on a T4 (15GB total, ~14GB available)
    bytes_per_element = 4  # for float32
    # Compute safe chunk size using the empirical safety factor.
    computed_chunk_size = compute_chunk_size(
        vocab, available_vram_gb, bytes_per_element, safety_factor=28
    )
    print(f"Computed maximum chunk size (with safety factor): {computed_chunk_size}")

    try:
        torch.cuda.reset_peak_memory_stats()
        X_large = torch.randn(
            bsz, qlen, hd, device="cuda", dtype=torch.float16, requires_grad=True
        )
        T_large = torch.randint(0, vocab, (bsz, qlen), device="cuda")
        linear_large = nn.Linear(hd, vocab, bias=True).cuda().half()
        loss_large = memory_efficient_cross_entropy_with_transform(
            X_large, linear_large, T_large, chunk_size=computed_chunk_size
        )
        print("Large input test loss:", loss_large.item())
        loss_large.backward()
        print_max_vram_usage("Large input test: ")
        print("Large input test completed without OOM.")
    except Exception as e:
        print("Large input test encountered an error:", e)


###########################################################
# 6. Test 3: Standard vs. Memory-Efficient Comparison
###########################################################
def test_standard_vs_memory_efficient():
    """
    Compare peak VRAM usage for the standard full-logits approach and the memory-efficient approach.
    We use an intermediate input size that is larger than the small test but small enough so that the standard
    approach does not run OOM.
    """
    print(
        "\nComparing standard vs. memory-efficient approaches on an intermediate input size:"
    )
    bsz, qlen, hd, vocab = 4, 2048, 1024, 32768
    X = torch.randn(
        bsz, qlen, hd, device="cuda", dtype=torch.float16, requires_grad=True
    )
    T = torch.randint(0, vocab, (bsz, qlen), device="cuda")
    linear = nn.Linear(hd, vocab, bias=True).cuda().half()

    def standard_ce_loss(input_feats, linear, targets):
        B, Q, D = input_feats.shape
        logits = linear(input_feats.view(-1, D)).float()  # full logits in float32
        loss = F.cross_entropy(logits, targets.view(-1))
        return loss

    # Measure standard approach.
    torch.cuda.reset_peak_memory_stats()
    loss_standard = standard_ce_loss(X, linear, T)
    loss_standard.backward()
    standard_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Standard approach loss: {loss_standard.item():.6f}")
    print_max_vram_usage("Standard approach: ")

    # Reset gradients.
    X.grad = None
    linear.weight.grad = None
    if linear.bias is not None:
        linear.bias.grad = None

    # Memory-efficient approach with a chunk size (e.g., 512 tokens).
    torch.cuda.reset_peak_memory_stats()
    chunk_size = 512
    loss_mem = memory_efficient_cross_entropy_with_transform(
        X, linear, T, chunk_size=chunk_size
    )
    loss_mem.backward()
    mem_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Memory-efficient approach loss: {loss_mem.item():.6f}")
    print_max_vram_usage("Memory-efficient approach: ")

    reduction_percent = (standard_vram - mem_vram) / standard_vram * 100
    print(f"Reduction in peak VRAM: {reduction_percent:.2f}%")


###########################################################
# 7. Main execution with cache clearing
###########################################################
if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_gradient_equivalence()
    test_large_input()
    test_standard_vs_memory_efficient()
    torch.cuda.empty_cache()
