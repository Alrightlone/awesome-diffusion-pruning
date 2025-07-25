# Awesome Diffusion Model Pruning

✨ A curated list of awesome resources, papers, and tools related to diffusion model pruning and efficiency.

## Introduction

Diffusion models have demonstrated state-of-the-art performance in generative tasks, creating high-fidelity images, audio, and more. However, their iterative nature and large parameter counts make them computationally expensive, hindering their deployment on resource-constrained devices. This challenge has catalyzed research into **Efficient Diffusion Models**.

Among the various efficiency techniques, **Model Pruning** stands out as a particularly promising direction.

### What is Diffusion Pruning?

**Pruning** is a model compression technique that aims to reduce model size and computational load by introducing **sparsity**. This is achieved by identifying and removing redundant or less important model weights, effectively **setting them to zero**. By eliminating these weights, we can significantly **decrease the storage footprint** and, with appropriate hardware or software support, **accelerate inference speed**.

In the context of diffusion models, pruning can be categorized into three main types:

1.  **Unstructured Pruning**: This method removes individual weights anywhere in the model based on a certain criterion (e.g., small magnitude). It offers the highest potential for compression and sparsity without being constrained by any structure. However, it results in a sparse weight matrix that often requires specialized libraries or hardware to achieve actual inference speed-ups, as standard dense matrix operations cannot be used.

2.  **Structured Pruning**: This method removes entire structured blocks of weights, such as neurons, channels, or even attention heads. For example, an entire filter in a convolutional layer might be zeroed out. This approach maintains a dense, regular structure in the weight matrices, which means it can directly lead to inference acceleration on off-the-shelf hardware (like GPUs and CPUs) without needing special handling. The trade-off is typically a larger drop in model accuracy for the same level of sparsity compared to unstructured pruning.

3.  **Semi-structured Pruning**: Also known as N:M sparsity, this is a hybrid approach that enforces a fine-grained sparsity pattern where N out of a small block of M weights are zeroed out (e.g., 2:4 sparsity). This method offers a compelling balance: it achieves higher compression rates than structured pruning while still mapping efficiently to modern hardware, such as NVIDIA's Ampere and later architectures that have dedicated support for sparse tensor operations.

### Pruning: An Orthogonal and Synergistic Approach

Model pruning fundamentally enhances efficiency by directly targeting architectural redundancy. By eliminating non-essential weights, pruning yields a dual benefit: it significantly **reduces the model's storage footprint** and **decreases the computational load (FLOPs)**, which can accelerate inference.

A key advantage of pruning is its **orthogonal nature** relative to other efficiency techniques. It can be synergistically combined with other methods for compounded gains:


- **Pruning + Quantization**: A pruned model can be subsequently quantized, applying low-precision numerics to an already smaller architecture for maximum model compression.
- **Pruning + Faster Samplers**: The reduced per-step cost of a pruned model, when paired with an advanced sampler (e.g., DPM-Solver) that requires fewer steps, results in a substantial decrease in total generation latency.
- **Pruning + Knowledge Distillation**: Pruning can simplify a large teacher model before distillation or be used to further compress a distilled student model.


This ability to act as a foundational optimization that complements other approaches makes pruning a uniquely powerful and versatile strategy for creating highly efficient diffusion models.

### Pruning Deserves Further Exploration

Given its unique ability to fundamentally reduce model complexity and its orthogonality to other popular methods, **pruning deserves further exploration**. It provides a direct path to smaller, more efficient diffusion models that are cheaper to store and serve. By creating sparsity, pruning opens up new possibilities for hardware acceleration and on-device deployment. As the demand for generative AI grows, a deeper understanding and development of advanced pruning techniques will be critical to making these powerful models accessible and sustainable for everyone.

This repository aims to curate the cutting-edge research and tools in this exciting domain. **Contributions are welcome!**
## Contents

- [Papers](#papers)
- [Tools & Libraries](#tools--libraries)
- [Tutorials & Documentation](#tutorials--documentation)

## Papers
| Title 📄                                                     | Method Summary 🛠️                                                                                                                              | Task 🎯                         | Pub or Date 🏛️ | Code 💻                                                              |
| :----------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------- | :------------- | :------------------------------------------------------------------- |
| [Structural Pruning for Diffusion Models](https://arxiv.org/abs/2305.10924) |  Diff-Pruning, an efficient compression method tailored for learning lightweight diffusion models from pre-existing ones, without the need for extensive re-training. The essence of Diff-Pruning is encapsulated in a Taylor expansion over pruned timesteps, a process that disregards non-contributory diffusion steps and ensembles informative gradients to identify important weights. | Unconditional Image Generation    | NIPS 2023      | [Github](https://github.com/VainF/Diff-Pruning)                        |
| [SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds](https://arxiv.org/abs/2306.00980) | We Present a generic approach that, for the first time, unlocks running text-to-image diffusion models on mobile devices in less than 2 seconds. We achieve so by introducing efficient network architecture and improving step distillation. Specifically, we propose an efficient UNet by identifying the redundancy of the original model and reducing the computation of the image decoder via data distillation. Further, we enhance the step distillation by exploring training strategies and introducing regularization from classifier-free guidance.| Text-to-Image                     | NIPS 2023    | [Github](https://github.com/snap-research/SnapFusion)      |
| [Effortless Efficiency: Low-Cost Pruning of Diffusion Models](https://arxiv.org/abs/2412.02852) |  In this work, we achieve low-cost diffusion pruning without retraining by proposing a model-agnostic structural pruning framework for diffusion models that learns a differentiable mask to sparsify the model. To ensure effective pruning that preserves the quality of the final denoised latent, we design a novel end-to-end pruning objective that spans the entire diffusion process. As end-to-end pruning is memory-intensive, we further propose time step gradient checkpointing, a technique that significantly reduces memory usage during optimization, enabling end-to-end pruning within a limited memory budget. Results on state-of-the-art U-Net diffusion models SDXL and diffusion transformers (FLUX) demonstrate that our method can effectively prune up to 20% parameters with minimal perceptible performance degradation, and notably, without the need for model retraining. We also showcase that our method can still prune on top of time step distilled diffusion models.| Text-to-Image                     | 2024.12    | Not provided yet |  
## Tools & Libraries
<!--- [Tool Name 1](http://example.com) - Brief description.-->
Coming soon!

## Tutorials & Articles
<!--- [Article Title 1](http://example.com) - Brief description.-->
Coming soon!