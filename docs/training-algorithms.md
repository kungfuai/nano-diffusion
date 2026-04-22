# Training Algorithms In This Repo

This document compares the major training algorithms implemented in this repo.

The goal is not to reproduce the papers in full detail. The goal is to answer practical questions:

- what problem each method is solving
- what happens conceptually during training
- what the model predicts
- how sampling works
- where the main code lives
- when you would reach for the method

## Overview

| Method | Main script | Core training idea | Typical prediction target | Sampling style |
| --- | --- | --- | --- | --- |
| DDPM | [`train_diffusion.py`](/home/zsi/projects/nano-diffusion/src/train_diffusion.py#L1) | Learn to denoise one timestep at a time | noise or related denoising target | multi-step reverse process |
| VDM | [`train_diffusion.py`](/home/zsi/projects/nano-diffusion/src/train_diffusion.py#L1) | Diffusion-style denoising with a different diffusion parameterization | denoising target from the VDM formulation | multi-step reverse process |
| CFM | [`train_cfm.py`](/home/zsi/projects/nano-diffusion/src/train_cfm.py#L1) | Learn the vector field of a continuous path from noise to data | velocity / conditional flow target | ODE integration |
| RF | [`train_rf.py`](/home/zsi/projects/nano-diffusion/src/train_rf.py#L1) | Rectified-flow variant of flow matching | velocity target with different time sampling | ODE integration, often simple Euler |
| MeanFlow | [`train_meanflow.py`](/home/zsi/projects/nano-diffusion/src/train_meanflow.py#L1) | Learn an average velocity over an interval instead of an instantaneous field | MeanFlow target from the MeanFlow identity | one-step or few-step updates |
| SiT | [`train_sit.py`](/home/zsi/projects/nano-diffusion/src/train_sit.py#L1) | Train on a general stochastic interpolant path | velocity, noise, or score | ODE integration over the interpolant |

## Shared High-Level View

All of these methods start from the same high-level goal:

- learn a model that can transform simple noise into data

The difference is how each method turns a full target sample into a supervised training problem.

For diffusion, the core training abstraction is:

- construct a one-step denoising problem for a sampled diffusion step

For flow-style methods, the broader training abstraction is:

- compute the training loss for a sampled state on a continuous path or interval

That is why diffusion in this repo now has a clean `prepare_step_supervision(...)` seam, while a more general cross-family abstraction would be:

- `prepare_step_supervision(...)`
- `compute_training_loss(...)`

Flow-style methods are usually better understood through `compute_training_loss(...)` or `training_losses(...)`, even when they are still supervising one local generation step at a time.

## DDPM / VDM

Main code:

- [`train_diffusion.py`](/home/zsi/projects/nano-diffusion/src/train_diffusion.py#L236)
- [`DiffusionTrainingConfig`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/config/diffusion_training_config.py#L9)
- [`create_diffusion_model_components()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/diffusion/diffusion_model_components.py#L35)
- [`training_loop()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/diffusion/diffusion_training_loop.py#L15)
- [`DDPM`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/diffusion/ddpm.py#L141)

### What happens conceptually

1. start from the final target sample `x_0`
2. sample one denoising step `t`
3. apply the forward noising process to produce `x_t`
4. train the model to predict the per-step target for that `t`
5. at sampling time, repeatedly denoise from pure noise to data

This is similar in spirit to autoregressive next-token training:

- a full object is turned into a per-step supervised prediction problem

### What the model predicts

Usually:

- noise

In this repo, that training-example construction happens through:

- [`BaseDiffusionAlgorithm.prepare_step_supervision()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/diffusion/base.py#L10)
- [`DDPM.prepare_step_supervision()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/diffusion/ddpm.py#L181)

### When to use it

Use DDPM or VDM when you want:

- the most standard diffusion training path
- a teaching-friendly baseline
- a stable, explicit denoising formulation

## CFM

Main code:

- [`train_cfm.py`](/home/zsi/projects/nano-diffusion/src/train_cfm.py#L90)
- [`ConditionalFlowMatcher`](/home/zsi/projects/nano-diffusion/src/train_cfm.py#L90)
- [`create_flow_matching_model_components()`](/home/zsi/projects/nano-diffusion/src/train_cfm.py#L658)
- [`training_loop()`](/home/zsi/projects/nano-diffusion/src/train_cfm.py#L465)

### What happens conceptually

1. sample clean data `x1`
2. sample source noise `x0`
3. sample a continuous time `t`
4. construct a point `x_t` on the path between `x0` and `x1`
5. compute the target vector field `v_t`
6. train the model output `u_theta(x_t, t)` to match that vector field
7. at sampling time, integrate the learned vector field from noise to data

### What the model predicts

- a learned velocity / vector-field estimate `u_theta(x_t, t)`

The core method is:

- [`ConditionalFlowMatcher.sample_location_and_conditional_flow()`](/home/zsi/projects/nano-diffusion/src/train_cfm.py#L207)

This is the step that constructs the training state and supervision for CFM.

For the base linear path in this repo,

```text
x_t = (1-t)x_0 + t x_1
```

so the local target comes from differentiating the path:

```text
v_t = d/dt x_t = x_1 - x_0
```

Then training is simply:

```text
u_theta(x_t, t) ~= v_t
```

### When to use it

Use CFM when you want:

- a continuous-time generative formulation
- ODE-based sampling
- a flow-style alternative to diffusion

## RF

Main code:

- [`train_rf.py`](/home/zsi/projects/nano-diffusion/src/train_rf.py#L1)
- [`RectifiedFlowMatcher`](/home/zsi/projects/nano-diffusion/src/train_rf.py#L42)

### What happens conceptually

1. follow the same overall path-based training idea as CFM
2. change the time-sampling strategy so training emphasizes more useful parts of the path
3. train the model to predict the rectified-flow velocity target
4. sample with a simple numerical integrator, often Euler

### What the model predicts

- a velocity field, like CFM

In this repo, RF is structurally a CFM-family variant rather than a completely separate training stack.

### When to use it

Use RF when you want:

- a flow-matching style method with a stronger practical bias
- a simple ODE-style sampler
- a rectified-flow path inspired by minRF

## MeanFlow

Main code:

- [`train_meanflow.py`](/home/zsi/projects/nano-diffusion/src/train_meanflow.py#L1)
- [`MeanFlowModelWrapper`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/meanflow/meanflow.py#L29)
- [`compute_meanflow_loss()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/meanflow/meanflow.py#L168)
- [`generate_samples_meanflow()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/meanflow/meanflow.py#L229)

### What happens conceptually

1. sample a pair of times `(r, t)` with `r <= t`
2. construct a noisy point `z_t` on the path between data and noise
3. define the instantaneous velocity target `v`
4. compute the model output `u_theta(z_t, r, t)`
5. compute the time derivative `du/dt` with JVP
6. build the MeanFlow target using the MeanFlow identity
7. train the model to match that interval-based target

### What the model predicts

Not the standard instantaneous velocity field.

Instead, the model output `u_theta(z_t, r, t)` predicts an average velocity over an interval, conditioned on both:

- `t`
- `r`

Intuition:

- standard flow matching learns the local instantaneous velocity at the current time
- MeanFlow learns the average update needed to move across the whole interval from `t` down to `r`
- that makes one-step and few-step generation a first-class training target, not just an afterthought at inference time

That is why this repo uses [`MeanFlowModelWrapper`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/meanflow/meanflow.py#L29), which adds extra conditioning for `delta = t - r`.

### MeanFlow identity

The implementation in [`compute_meanflow_loss()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/meanflow/meanflow.py#L168) uses:

```text
u_theta(z_t, r, t) = v(z_t, t) - (t-r) * d/dt u_theta(z_t, r, t)
```

where:

- `v(z_t, t)` is the instantaneous velocity field
- `u_theta(z_t, r, t)` is the learned interval-conditioned average velocity field
- `d/dt` is the total derivative along the trajectory, with `r` held fixed

One college-calculus derivation is:

```text
u_theta(z_t, r, t) := 1/(t-r) * integral_r^t v(z_s, s) ds
```

Equivalently,

```text
(t-r) * u_theta(z_t, r, t) = integral_r^t v(z_s, s) ds
```

Now differentiate both sides with respect to `t`, holding `r` fixed:

- right-hand side: by the Fundamental Theorem of Calculus,
  `d/dt integral_r^t v(z_s, s) ds = v(z_t, t)`
- left-hand side: by the product rule,
  `d/dt [(t-r)u] = u + (t-r) * d/dt u`

Set them equal and rearrange:

```text
u_theta + (t-r) * d/dt u_theta = v
u_theta = v - (t-r) * d/dt u_theta
```

That is the identity used to form the training target.

This is a useful general pattern, not a MeanFlow-specific coincidence:

- start from a multi-step generation goal
- derive a local supervised training target for one sampled turn
- train on those local turns, then chain them again at sampling time

What changes across algorithms is how that local supervision is derived:

- diffusion derives it from the forward noising / reverse denoising process
- CFM and RF derive it from the transport path and vector field
- SiT derives it from the interpolant path and prediction parameterization
- MeanFlow derives it especially explicitly by differentiating an interval-level identity

### Loss

The code builds the target:

```text
u_tgt = v - (t-r) * d/dt u_theta
```

and then optimizes an adaptive weighted L2 loss on:

```text
error = u_theta - stopgrad(u_tgt)
```

See:

- [`compute_meanflow_loss()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/meanflow/meanflow.py#L211)
- [`adaptive_l2_loss()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/meanflow/meanflow.py#L160)

### Sampling

At sampling time, the model is used directly as an interval update rule:

```text
z_r = z_t - (t-r) * u_theta(z_t, r, t)
```

This means one sampling step is:

1. start from the current state `z_t`
2. choose the next earlier time `r`
3. predict the average velocity `u_theta(z_t, r, t)`
4. move directly to `z_r`

This repo implements that in [`generate_samples_meanflow()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/meanflow/meanflow.py#L229).

Special cases:

- one-step generation: choose `t=1, r=0`, so `x = z - u_theta(z, 0, 1)`
- few-step generation: chain a small number of interval updates
- multi-step generation: use a finer time grid and repeatedly apply the same update rule

MeanFlow is attractive because it supports:

- one-step generation
- few-step generation

while still remaining compatible with chained multi-step sampling.

### When to use it

Use MeanFlow when you want:

- fast generation
- one-step or few-step generation experiments
- a model trained explicitly for that regime

## SiT

Main code:

- [`train_sit.py`](/home/zsi/projects/nano-diffusion/src/train_sit.py#L1)
- [`Transport`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/sit/transport.py#L14)
- [`create_path()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/sit/interpolant.py#L99)
- [`euler_ode_sample()`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/sit/transport.py#L113)

### What happens conceptually

1. choose an interpolant path
   - linear
   - trigonometric / GVP
   - variance-preserving
2. sample clean data `x1`
3. sample source noise `x0`
4. sample a time `t`
5. construct the interpolated point `x_t = alpha_t x1 + sigma_t x0`
6. choose what the model should predict
   - velocity
   - noise
   - score
7. construct the per-turn target implied by that prediction type
8. compute the corresponding training loss for that sampled turn
9. at sampling time, chain many such local turns by numerically integrating over the interpolant path

### What the model predicts

One of:

- velocity
- noise
- score

This flexibility is one reason the central abstraction is:

- [`Transport.training_losses(...)`](/home/zsi/projects/nano-diffusion/src/nanodiffusion/sit/transport.py#L38)

rather than an explicitly shared `(inputs, targets)` interface.

This is still a per-turn supervised training problem.
The important nuance is that the model output is not always the exact update used by the sampler:

- if the model predicts `velocity`, the sampler can integrate that field directly
- if the model predicts `noise` or `score`, the sampler converts that prediction into the implied drift/update using the interpolant path coefficients

### When to use it

Use SiT when you want:

- an interpolant-based transformer training setup
- flexibility in path choice and prediction target
- a bridge between DiT-style architectures and flow/interpolant training

## How The Methods Differ

### Diffusion vs flow-style methods

Diffusion:

- training is usually framed as a one-step denoising problem
- sampling is a reverse denoising chain

Flow-style methods:

- training is usually framed as learning a field or interval update on a continuous path
- sampling is usually ODE-style integration or interval updates

### Instantaneous field vs interval-based update

CFM / RF / many SiT settings:

- learn a local field at time `t`

MeanFlow:

- learn an interval-conditioned average update over `[r, t]`

### Multi-step vs fast generation

DDPM / VDM:

- naturally multi-step

CFM / RF / SiT:

- ODE integration, potentially fewer steps depending on quality needs

MeanFlow:

- explicitly designed for one-step or few-step generation

## Practical Recommendation

If you are learning:

- start with DDPM
- then read CFM
- then compare RF and SiT
- then read MeanFlow when you care about fast generation

If you are optimizing for practical speed:

- MeanFlow is the most directly aligned with one-step or few-step inference
- RF and SiT are also important if you want flow-style generation without standard diffusion sampling

If you are optimizing for conceptual clarity:

- DDPM and CFM are the clearest anchor points in this repo
