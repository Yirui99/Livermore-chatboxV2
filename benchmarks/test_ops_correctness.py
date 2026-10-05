"""
Numerical Correctness Tests for custom ops:
  - ops.softmax vs torch.nn.functional.softmax
  - ops.layer_norm vs torch.nn.functional.layer_norm
  - ops.gelu vs torch.nn.functional.gelu
  - ops.matmul vs torch.matmul

Reports max absolute error, max relative error, and allclose pass/fail.
"""
import torch
import torch.nn.functional as F
import sys, os
from livermore.scratch import ops

def test_softmax():
    print("=" * 70)
    print("TEST: ops.softmax vs torch.nn.functional.softmax")
    print("=" * 70)

    test_cases = [
        ("Normal fp32",       torch.randn(2, 8, 128, 128)),
        ("Normal fp16",       torch.randn(2, 8, 128, 128).half()),
        ("Large values",      torch.randn(2, 8, 64, 64) * 100),
        ("Very large values", torch.randn(2, 4, 32, 32) * 1000),
        ("Small values",      torch.randn(2, 4, 32, 32) * 1e-6),
        ("Mixed sign",        torch.randn(2, 4, 64, 64) * 50 - 25),
    ]

    for name, x in test_cases:
        custom = ops.softmax(x)
        native = F.softmax(x.float(), dim=-1).to(x.dtype)

        max_abs = (custom.float() - native.float()).abs().max().item()
        max_rel = ((custom.float() - native.float()).abs() / (native.float().abs() + 1e-12)).max().item()
        close = torch.allclose(custom.float(), native.float(), atol=1e-5, rtol=1e-5)

        # Check for NaN/Inf
        has_nan = custom.isnan().any().item()
        has_inf = custom.isinf().any().item()

        status = "✅ PASS" if (close and not has_nan and not has_inf) else "❌ FAIL"
        print(f"  {name:<25} | max_abs_err={max_abs:.2e} | max_rel_err={max_rel:.2e} | nan={has_nan} inf={has_inf} | {status}")

    print()


def test_layer_norm():
    print("=" * 70)
    print("TEST: ops.layer_norm vs torch.nn.functional.layer_norm")
    print("=" * 70)

    test_cases = [
        ("Normal fp32 (B=2,T=64,D=256)",   torch.randn(2, 64, 256)),
        ("Normal fp16 (B=2,T=64,D=256)",   torch.randn(2, 64, 256).half()),
        ("Large values",                    torch.randn(2, 32, 128) * 100),
        ("Small values",                    torch.randn(2, 32, 128) * 1e-5),
    ]

    for name, x in test_cases:
        D = x.shape[-1]
        gamma = torch.ones(D, dtype=x.dtype)
        beta = torch.zeros(D, dtype=x.dtype)

        custom = ops.layer_norm(x, gamma, beta, eps=1e-5)
        native = F.layer_norm(x.float(), (D,), weight=gamma.float(), bias=beta.float(), eps=1e-5).to(x.dtype)

        max_abs = (custom.float() - native.float()).abs().max().item()
        max_rel = ((custom.float() - native.float()).abs() / (native.float().abs() + 1e-12)).max().item()
        close = torch.allclose(custom.float(), native.float(), atol=1e-5, rtol=1e-5)

        status = "✅ PASS" if close else "❌ FAIL"
        print(f"  {name:<35} | max_abs_err={max_abs:.2e} | max_rel_err={max_rel:.2e} | {status}")

    print()


def test_gelu():
    print("=" * 70)
    print("TEST: ops.gelu vs torch.nn.functional.gelu")
    print("=" * 70)

    test_cases = [
        ("Exact GELU fp32",         torch.randn(4, 128, 256), False),
        ("Approx GELU fp32",        torch.randn(4, 128, 256), True),
        ("Exact GELU large values", torch.randn(4, 64, 128) * 10, False),
        ("Approx GELU large values",torch.randn(4, 64, 128) * 10, True),
    ]

    for name, x, approx in test_cases:
        custom = ops.gelu(x, approximate=approx)
        native = F.gelu(x, approximate="tanh" if approx else "none")

        max_abs = (custom - native).abs().max().item()
        max_rel = ((custom - native).abs() / (native.abs() + 1e-12)).max().item()
        close = torch.allclose(custom, native, atol=1e-6, rtol=1e-6)

        status = "✅ PASS" if close else "❌ FAIL"
        print(f"  {name:<30} | max_abs_err={max_abs:.2e} | max_rel_err={max_rel:.2e} | {status}")

    print()


def test_matmul():
    print("=" * 70)
    print("TEST: ops.matmul vs torch.matmul")
    print("=" * 70)

    test_cases = [
        ("2D (128, 64) @ (64, 256)",       torch.randn(128, 64),      torch.randn(64, 256)),
        ("3D (2, 128, 64) @ (2, 64, 256)", torch.randn(2, 128, 64),   torch.randn(2, 64, 256)),
        ("4D (2, 4, 64, 64) @ (2, 4, 64, 64)", torch.randn(2, 4, 64, 64), torch.randn(2, 4, 64, 64)),
    ]

    for name, A, B in test_cases:
        custom = ops.matmul(A, B)
        native = torch.matmul(A, B)

        max_abs = (custom - native).abs().max().item()
        close = torch.allclose(custom, native, atol=1e-5, rtol=1e-5)

        status = "✅ PASS" if close else "❌ FAIL"
        print(f"  {name:<45} | max_abs_err={max_abs:.2e} | {status}")

    print()


def test_fp16_overflow_protection():
    print("=" * 70)
    print("TEST: FP16 overflow protection in ops.softmax")
    print("=" * 70)

    # Construct extreme fp16 logits that would overflow naive softmax
    # fp16 max is ~65504; exp(11) ≈ 59874 which is near the limit
    x_extreme = torch.tensor([[500.0, 600.0, 700.0, 800.0]], dtype=torch.float16)

    # Naive softmax (no max-subtraction) would overflow in fp16
    print(f"  Input (fp16): {x_extreme}")

    # Custom ops.softmax (has upcast + max-subtraction protection)
    custom_out = ops.softmax(x_extreme)
    has_nan_custom = custom_out.isnan().any().item()
    has_inf_custom = custom_out.isinf().any().item()

    # Naive implementation WITHOUT protection
    try:
        naive_exp = torch.exp(x_extreme)  # This will overflow in fp16
        naive_out = naive_exp / naive_exp.sum(dim=-1, keepdim=True)
        has_nan_naive = naive_out.isnan().any().item()
        has_inf_naive = naive_out.isinf().any().item()
    except Exception as e:
        has_nan_naive = True
        has_inf_naive = True
        naive_out = None

    print(f"  Naive softmax (no protection):   nan={has_nan_naive}  inf={has_inf_naive}  → {'❌ BROKEN' if (has_nan_naive or has_inf_naive) else '✅ OK'}")
    print(f"  ops.softmax (with protection):   nan={has_nan_custom} inf={has_inf_custom} → {'✅ SAFE' if (not has_nan_custom and not has_inf_custom) else '❌ BROKEN'}")
    if not has_nan_custom and not has_inf_custom:
        print(f"  Output: {custom_out}")
        print(f"  Sum:    {custom_out.sum().item():.6f} (should be ~1.0)")

    print()


if __name__ == "__main__":
    test_softmax()
    test_layer_norm()
    test_gelu()
    test_matmul()
    test_fp16_overflow_protection()

    print("=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
