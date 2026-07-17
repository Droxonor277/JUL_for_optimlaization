# B0B36JUL – Julia for Optimization and Learning

This repository contains solutions to programming assignments from the course **B0B36JUL – Julia for Optimization and Learning** at the Faculty of Electrical Engineering, Czech Technical University in Prague.

The assignments are implemented in **Julia** and focus on numerical methods, optimization, and machine learning.

## Repository Contents

### Root-Finding Methods

Implementation of the **bisection** and **regula falsi** methods for finding roots of continuous functions.

The solution demonstrates:

* multiple dispatch,
* numerical tolerances,
* stopping criteria,
* input validation.

### Multiple Local Minima

Implementation of a multi-start projected gradient method for finding multiple solutions in a bounded domain.

The assignment includes:

* projected gradient descent,
* box constraints,
* duplicate-solution filtering,
* the non-convex Griewank function.

### Support Vector Machine

Implementation of a linear SVM classifier trained by solving its dual problem using coordinate descent.

The project covers:

* construction of the dual matrix,
* coordinate-wise optimization,
* reconstruction of classifier weights,
* classification of the Iris dataset.

## Technologies Used

* Julia
* LinearAlgebra
* Statistics
* numerical optimization methods

## Note

This repository contains student solutions created for educational purposes. The code is intended for studying Julia, numerical algorithms, optimization, and basic machine-learning methods.
