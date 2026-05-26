# JAX 101

In this section, we will cover the basics to advanced of new framework called **JAX**. 
JAX is a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning.

## Feature of JAX:

1.  **Familiar API**: JAX provides a familiar Numpy-Sytle API for ease of adoption by researchers and engineers.
2.  **Transformations**: JAX includes composable function transformations for compilation, batching, automatic differentiation, and parallelization.
3.  **Run Anywhere**: The same code executes on multiple backends, including CPU, GPU, & TPU.

## Installation

Using JAX requires installing two packages: 
1. `jax`, which is pure Python and cross-platform, and 
2. `jaxlib` which contains compiled binaries, and requires different builds for different operating systems and accelerators.

For general purpose we can install `JAX` using the below command:
```commandline
pip install -U JAX
```
The above code is generally for CPU-only(Linux, MacOs, Windows)

For `GPU(Nvidia, CUDA13)`, we can use:

```commandline
pip install -U "jax[cuda13]"

```

For `GPU(AMD, ROCm)` we can use :

```commandline
pip install -U "jax[rocm7-local]"

```

And finally for `TPU(Google Cloud TPU VM)`, we can use:

```commandline
pip install -U "jax[tpu]"

```

For more information about installation of JAX follow this [Link](https://docs.jax.dev/en/latest/installation.html).

