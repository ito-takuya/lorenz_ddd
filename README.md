# lorenz_ddd – ML-based data-driven discovery tutorials using the Lorenz system
#### Takuya Ito
07/07/2025

This repository contains a set of python notebook tutorials on ML-based data-driven discovery.

The experimental demonstration is to infer the underlying differential equations of the Lorenz system from generated data using various ML models.

$$\dot{x} = \sigma (y - x)$$

$$\dot{y} = x ( \rho - z) - y$$

$$\dot{z} = xy - \beta z$$

The general task is, given the values of the Lorenz system $x(t)$, $y(t)$, and $z(t)$, to learn a function $f$ such that $f$ maps to the instantaneous rate of change:

$$ \dot{x} = f_x \big( x(t),y(t),z(t) \big)$$
$$ \dot{y} = f_y \big( x(t),y(t),z(t) \big)$$
$$ \dot{z} = f_z \big( x(t),y(t),z(t) \big)$$

The tutorials included are:
1. Polynomial regression (Notebook2) -- see Brunton et al. (2015) *PNAS*
2. Multilayer perceptrons (Notebook3)
3. Small transformers (Notebook4)

