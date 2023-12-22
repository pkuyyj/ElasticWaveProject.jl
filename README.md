# Elastic Wave Project

Final project of Yuejiang Yu

## Introduction to Elastic Wave Propagation in 3D

This document provides an overview of the simulation of elastic waves using 3D wave propagation equations.

### Governing Equations

The governing equations for the elastic wave propagation are as follows:

```plaintext
ρ ∂v_z/∂t = ∂S_zz/∂z + ∂S_xz/∂x + ∂S_yz/∂y
ρ ∂v_x/∂t = ∂S_xz/∂x + ∂S_zz/∂z + ∂S_xy/∂y
ρ ∂v_y/∂t = ∂S_yz/∂y + ∂S_xy/∂x + ∂S_yy/∂z
∂σ_xx/∂t = λ (∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z) + 2μ ∂v_x/∂x
∂σ_yy/∂t = λ (∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z) + 2μ ∂v_y/∂y
∂σ_zz/∂t = λ (∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z) + 2μ ∂v_z/∂z
∂σ_xz/∂t = μ(∂v_z/∂x + ∂v_x/∂z)
∂σ_xy/∂t = μ(∂v_y/∂x + ∂v_x/∂y)
∂σ_yz/∂t = μ(∂v_z/∂y + ∂v_y/∂z)
```

### Variables Description

Here is a table describing each variable used in the equations:

| Variable | Description                                       |
|----------|---------------------------------------------------|
| v_x, v_y, v_z | Velocity of the wave field in the X, Y and Z direction|
| σ_xx, σ_yy, σ_zz, σ_xz, σ_yz, σ_xy | Stress tensor           |
| ρ        | Density of the media                              |
| λ        | Lamé's first parameter                            |
| μ        | Shear modulus                                     |


### Use damping factor to prevent gradient explosion

Actually, the energy carried by elastic wave will transfer to other energy such as heat during propagation, so a damping factor (or two) is needed.

Here is the pseudo code of Stress update in practice:

```julia
Sxx_new = damping_sxx * Sxx_old + dt * (λ * (∂Vx/∂x + ∂Vy/∂y + ∂Vz/∂z) + 2μ * ∂Vx/∂x)
Syy_new = damping_sxx * Syy_old + dt * (λ * (∂Vx/∂x + ∂Vy/∂y + ∂Vz/∂z) + 2μ * ∂Vy/∂y)
Szz_new = damping_sxx * Szz_old + dt * (λ * (∂Vx/∂x + ∂Vy/∂y + ∂Vz/∂z) + 2μ * ∂Vz/∂z)
Sxy_new = damping_sxy * Sxy_old + dt * μ * ((∂Vy/∂x) + (∂Vx/∂y))
Sxz_new = damping_sxy * Sxz_old + dt * μ * ((∂Vz/∂x) + (∂Vx/∂z))
Syz_new = damping_sxy * Syz_old + dt * μ * ((∂Vz/∂y) + (∂Vy/∂z))
```

We use two damping factors `damping_sxx` and `damping_sxy` which stands for the damping factors of normal stress and shear stress. In practice, shear stress tensor is more likely to explode. Therefore `damping_sxy` is set to `0.95`.



## Usage

### Installation

```bash
bash script/install.sh
```

### Slurm submission on Piz Daint

First, change const variable `USE_GPU = false` to `true` and add `elastic3D()` to the end of `wave3D_multixpu.jl`.

Then run the following command:

```bash
sbatch script/wavempi.sh
```

After about 2 minutes, the output should be in `./viz3D_out/`. You can see `wave3D.gif` in this folder.

### Testing

Testing is automatically done using Github's CI/CD.

### 3D contour visualization

We need another julia environment with `GLMakie` installed. Then run:
```bash
julia src/visualize
```

## Results

We run elastic wave simulation in 3D on 4 GPUs. Time step is set to `20k` with `120^3` grids. Below is the result:

![wave_3D](./docs/wave_3D.gif)

The main difference between elastic wave and acoustic wave is that it has P-waves (Primary waves) and S-waves (Secondary waves). We can clearly see that there are two waves with different speed in the gif plot.

The 3D stress are also shown in this animation:

![wave_3D_contour](./docs/wave_3D_contour.gif)

## Discussion, Conclusion and Outlook
The project confirms the complex nature of elastic wave, demonstrating how it can be reduced to simpler wave equations under certain conditions. The results have implications for understanding natural systems and enhancing seismology practices. Future work may explore more complex initialization state, larger system.