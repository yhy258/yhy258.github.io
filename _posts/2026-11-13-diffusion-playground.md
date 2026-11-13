---
title: "A Diffusion Model You Can Run in This Page"
kicker: "Sample post"
description: "A toy diffusion model on 2D points, with an exact score function instead of a neural network. Also a tour of what a post on this blog can contain."
tags: [diffusion, generative-models]
scripts:
  - /assets/js/demos/diffusion.js
---

This is a sample post. It shows what a post on this blog can contain: math, code, footnotes, notes, and
interactive figures. Edit it or delete it; it lives in `_posts/2026-11-13-diffusion-playground.md`.
{: .note}

## The demo

Drag the slider to add noise to the data. Then press **Generate from noise** to run the process backwards,
from pure noise to samples.

<figure class="widget" data-demo="diffusion">
  <figcaption><b>Figure 1.</b> 700 points under a variance-preserving diffusion. The slider runs the forward
  process in closed form. The button integrates the reverse process from \(t = 1\) to \(t = 0\) with the exact
  score of the data distribution.</figcaption>
</figure>

## What is being computed

### The forward process

The forward process mixes a clean sample $$x_0$$ with Gaussian noise[^ddpm]:

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I).
$$

Here $$\bar\alpha_t = \exp\!\big(-\!\int_0^t \beta(s)\, ds\big)$$, with a linear schedule
$$\beta(t)$$ from 0.1 to 20. The slider's readout shows the two coefficients as "signal" and "noise".

### The reverse process

Sampling runs time backwards. Both samplers in the figure need only the score
$$\nabla_x \log p_t(x)$$[^sde]:

$$
\begin{aligned}
\text{SDE:}\quad dx &= \big[-\tfrac{1}{2}\beta(t)\, x - \beta(t)\, \nabla_x \log p_t(x)\big]\, dt + \sqrt{\beta(t)}\, d\bar{w}, \\
\text{ODE:}\quad dx &= -\tfrac{1}{2}\beta(t)\, \big[x + \nabla_x \log p_t(x)\big]\, dt.
\end{aligned}
$$

A real diffusion model learns the score with a neural network. Here the data is a mixture of Gaussians with
means $$\mu_k$$ and standard deviation $$s$$, so $$p_t$$ is again a mixture of Gaussians and its score has a
closed form:

$$
\nabla_x \log p_t(x) = \sum_k r_k(x)\, \frac{\sqrt{\bar\alpha_t}\,\mu_k - x}{v_t},
\qquad v_t = \bar\alpha_t s^2 + 1 - \bar\alpha_t,
$$

where $$r_k(x)$$ is the posterior probability that $$x$$ came from component $$k$$. Tick **Show score field**
to see this vector field change with $$t$$.

## Code

Code blocks are highlighted and have a copy button. One reverse step of the SDE sampler:

```python
def reverse_step(x, t, dt, score):
    """One Euler-Maruyama step of the reverse SDE, from t to t - dt."""
    b = beta(t)
    drift = 0.5 * b * x + b * score(x, t)
    return x + drift * dt + (b * dt) ** 0.5 * torch.randn_like(x)
```

<details markdown="1">
<summary>How to add an interactive figure to a post</summary>

1. Write the figure as a script in `assets/js/demos/`. It finds its own element in the page and fills it.
2. Put the element in the post: `<figure class="widget" data-demo="..."></figure>`.
3. List the script in the post's front matter under `scripts:`.

</details>

## Writing a post

| To write | Type |
| --- | --- |
| Inline or display math | `$$ ... $$` (within a line, or as its own paragraph) |
| A footnote | `[^name]` in the text, `[^name]: ...` at the end |
| A note box | a paragraph followed by `{: .note}` |
| A figure | `<figure> <img src="..."> <figcaption>...</figcaption> </figure>` |

[^ddpm]: Ho, Jain, and Abbeel. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS 2020.
[^sde]: Song et al. [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456). ICLR 2021.
