# From Global Objectives To Local Stepwise Supervision

This note captures a useful pattern that shows up across diffusion, flow matching, MeanFlow, and reinforcement learning:

- start from a sequential multi-step goal
- derive a local training signal for one sampled step
- train on local steps
- chain those local predictions again at inference or control time

The details differ a lot across algorithms. The pattern is still real.

## Core idea

Many useful learning problems are naturally global:

- generate a full image from noise
- transport one distribution into another
- maximize total future reward over a trajectory

Training directly on the whole trajectory end to end is often expensive, unstable, or mathematically awkward.

So the standard move is:

1. write down the full multi-step process
2. identify the local quantity that, when chained over many steps, realizes that process
3. derive supervision for that local quantity
4. train a model on randomly sampled local steps

That is the common theme.

## Flow matching

For standard CFM, choose a continuous path `x_t` connecting `x_0` to `x_1`:

```text
x_t = x_0 + integral_0^t v_s ds
```

Then:

```text
x_1 - x_0 = integral_0^1 v_s ds
```

and the local target comes from differentiating the path:

```text
d/dt x_t = v_t
```

In the base linear path used in this repo:

```text
x_t = (1-t) x_0 + t x_1
v_t = d/dt x_t = x_1 - x_0
```

Then train a model `u_theta(x_t, t)` to match `v_t`.

Sampling later integrates the learned local field back into a full trajectory.

## MeanFlow

MeanFlow keeps the same global-to-local pattern, but the local target is derived differently.

Instead of learning the instantaneous velocity directly, it learns an interval-conditioned average velocity:

```text
u_theta(z_t, r, t)
```

One useful way to define it is:

```text
u_theta(z_t, r, t) := 1/(t-r) * integral_r^t v(z_s, s) ds
```

Differentiate:

```text
(t-r) u_theta = integral_r^t v(z_s, s) ds
u_theta + (t-r) * d/dt u_theta = v
u_theta = v - (t-r) * d/dt u_theta
```

That identity gives the local target used during training.

So MeanFlow also starts from a multi-step generation object and converts it into local supervision. It just does so through an interval identity instead of a plain path derivative.

## Diffusion

Diffusion follows the same high-level pattern, but the derivation is usually probabilistic rather than calculus-first.

Start from the full forward corruption process, for example:

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
```

Then sample one timestep `t`, construct the corrupted state `x_t`, and derive a target for the reverse denoising step:

- noise
- `x_0`
- velocity

The local supervision comes from the conditional structure of the forward and reverse processes, not mainly from differentiating a transport path.

So diffusion still fits the same pattern:

- global noising / denoising process
- local one-step supervision
- repeated reverse steps at sampling time

## Reinforcement learning

RL is related, but the mechanism is different again.

The global objective is something like:

```text
maximize expected total future reward
```

or, in continuous notation,

```text
G_t = integral_t^T r(s) ds
```

The practical per-step training signal usually does not come from differentiating that reward integral with respect to time.

Instead, RL typically derives local learning signals through:

- Bellman recursion
- temporal-difference targets
- policy gradient identities

Examples:

```text
V(s_t) = E[r_t + gamma V(s_{t+1})]
Q target = r_t + gamma max_a Q(s_{t+1}, a)
policy gradient ~ grad log pi(a_t | s_t) * A_t
```

So RL fits the same broad pattern:

- global trajectory objective
- local stepwise training signal

But the local signal is derived from recursion or gradient identities, not usually from differentiating the trajectory integral itself.

## Similarity and difference

What is shared:

- a multi-step objective is converted into local trainable signals
- the model is trained on sampled local steps
- a full trajectory is recovered by chaining those local decisions or predictions

What differs:

- flow matching: local target often comes from differentiating a chosen path
- MeanFlow: local target comes from differentiating an interval-level identity
- diffusion: local target comes from probabilistic conditioning
- RL: local target comes from Bellman recursion or policy-gradient identities

## Why this matters for this repo

This perspective is useful when designing training interfaces.

A good internal abstraction often looks like:

- prepare or compute supervision for one sampled step
- train the model on that local step
- keep sampling logic separate, where those local predictions are chained into a full generation process

That is why abstractions like:

- `prepare_step_supervision(...)`
- `compute_training_loss(...)`
- `sample(...)`

are natural seams for this codebase.
