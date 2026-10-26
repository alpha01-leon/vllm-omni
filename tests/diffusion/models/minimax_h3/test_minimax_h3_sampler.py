# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Coverage for the MiniMax-H3 request-local solvers.

The `res_multistep` solver is reachable on the UNIFORM sigma schedule --
the guard in the pipeline only rejects it for a FIXED DISTILLED schedule
(`base_schedule is not None`), which is what the distilled Turbo releases
pin. Until now nothing exercised the uniform path, so the solver could
regress without any test noticing: the only `res_multistep` assertion in the
tree (`test_vdnh3.py`) covers a different guard entirely.

These tests pin the uniform-schedule behaviour: the coefficient shape (the
first and last steps degenerate to Euler), the finiteness of every step, and
that the two solvers agree on where the trajectory ends up.

`res_multistep` is NOT compared against a reference implementation here.
It has no ComfyUI-side reference to compare to on this machine, so the
assertions are about internal consistency and stability, not about matching
an external oracle. The pipeline's guard message says as much.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

NUM_STEPS = 8
SHIFT = 12.0
STATE_SHAPE = (4, 96)


def _uniform_sigmas(num_steps: int = NUM_STEPS, shift: float = SHIFT) -> list[float]:
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    return minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=shift)


def _run_solver(name: str, sigmas: list[float], *, seed: int = 0) -> list[torch.Tensor]:
    """Drive one solver over the whole ladder, returning each step's output."""
    from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver

    generator = torch.Generator().manual_seed(seed)
    solver = create_h3_sample_solver(name, sigmas)
    state = torch.randn(*STATE_SHAPE, generator=generator, dtype=torch.float32)
    outputs = []
    for step in range(len(sigmas) - 1):
        # A constant x0 prediction: the solver's arithmetic is what is under
        # test, not the model's denoiser.
        denoised = torch.full(STATE_SHAPE, 0.25, dtype=torch.float32)
        state = solver.step(state, denoised, step)
        outputs.append(state)
    return outputs


def test_uniform_sigma_ladder_shape():
    """The uniform ladder is one sigma more than the number of steps."""
    sigmas = _uniform_sigmas()
    assert len(sigmas) == NUM_STEPS + 1
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == pytest.approx(0.0)
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


def test_res_multistep_degenerates_to_euler_at_both_ends():
    """The first and last steps fall back to Euler, so they carry no RES coefficient.

    That is not an implementation detail to stay quiet about: it is why the
    solver's history is only consulted on the interior steps.
    """
    from vllm_omni.diffusion.models.minimax_h3.sampling import res_multistep_coeffs

    coeffs = res_multistep_coeffs(_uniform_sigmas())
    assert len(coeffs) == NUM_STEPS
    assert coeffs[0] is None
    assert coeffs[-1] is None
    assert all(coeff is not None for coeff in coeffs[1:-1])


def test_res_multistep_steps_are_finite_and_shape_preserving():
    """Every step of the uniform ladder stays finite and keeps the latent shape."""
    outputs = _run_solver("res_multistep", _uniform_sigmas())
    assert len(outputs) == NUM_STEPS
    for step, out in enumerate(outputs):
        assert out.shape == STATE_SHAPE, f"step {step} changed the latent shape"
        assert torch.isfinite(out).all(), f"step {step} produced non-finite values"


def test_res_multistep_and_euler_agree_on_the_first_and_last_step():
    """Both ends degenerate to Euler, so the two solvers must match there exactly.

    This is the strongest internal-consistency statement available without a
    reference implementation: the first and last steps are literally the
    Euler formula, so any difference would mean the RES path is leaking into
    the degenerate steps.
    """
    sigmas = _uniform_sigmas()
    euler = _run_solver("euler", sigmas)
    res = _run_solver("res_multistep", sigmas)
    torch.testing.assert_close(res[0], euler[0], rtol=0, atol=0)
    torch.testing.assert_close(res[-1], euler[-1], rtol=0, atol=0)


def test_res_multistep_stays_close_to_euler_over_the_trajectory():
    """The two solvers must not diverge into different places.

    A second-order solver is a refinement, not a different sampler, so on the
    same inputs it has to track Euler. The bound is deliberately loose: this
    pins "same trajectory", not "same numbers".
    """
    sigmas = _uniform_sigmas()
    euler = _run_solver("euler", sigmas)
    res = _run_solver("res_multistep", sigmas)
    for step, (a, b) in enumerate(zip(res, euler, strict=True)):
        rel = (a - b).norm() / b.norm().clamp(min=1e-12)
        assert rel < 0.5, f"step {step}: res_multistep diverged from euler (rel {rel:.4f})"


def test_res_multistep_rejects_a_non_decreasing_ladder():
    """The coefficient builder refuses a schedule that is not strictly decreasing."""
    from vllm_omni.diffusion.models.minimax_h3.sampling import res_multistep_coeffs

    with pytest.raises(ValueError, match="strictly decreasing"):
        res_multistep_coeffs([1.0, 0.5, 0.5, 0.0])


def test_res_multistep_rejects_an_out_of_range_ladder():
    from vllm_omni.diffusion.models.minimax_h3.sampling import res_multistep_coeffs

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        res_multistep_coeffs([1.0, 0.5, 1.5])


def test_solver_rejects_out_of_order_steps():
    """A solver's steps must be consumed once, in order."""
    from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver

    solver = create_h3_sample_solver("res_multistep", _uniform_sigmas())
    state = torch.randn(*STATE_SHAPE, dtype=torch.float32)
    denoised = torch.full(STATE_SHAPE, 0.25, dtype=torch.float32)
    with pytest.raises(ValueError, match="consumed once, in order"):
        solver.step(state, denoised, 1)
