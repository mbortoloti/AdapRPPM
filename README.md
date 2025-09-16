

**An adaptative proximal point method for nonsmooth and nonconvex optimization on Hadamard manifolds**

Vitaliano Amaral, Marcio Bortoloti, Jurandir O. Lopes, Gilson Silva


**Abstract** This paper addresses a class of nonsmooth and nonconvex optimization problems defined on complete Riemannian manifolds. The objective function has a composite structure, combining convex, differentiable, and lower semicontinuous terms, thereby generalizing the classical framework of difference-of-convex programming. Motivated by recent advances in proximal point methods in both Euclidean and Riemannian settings, we propose two
variants of the proximal point method for solving this class of problems. The first variant requires prior knowledge of the Lipschitz constant of the gradient of the smooth part, making it suitable when this parameter can be readily computed. The second variant, in contrast, does not require such knowledge, thus broadening its applicability. We analyze the complexity of both approaches, establish their convergence, and illustrate their effectiveness through numerical experiments.

To run the adap-RPPM algorithm, you must provide the objective function and its corresponding Euclidean gradients (or subgradients).

Please see the file example.jl for a functional code example. It solves the following optimization problem:

Minimize
$f(X) = g_1(X) + g_2(X) - h(X)$

where:

* The function $f$ is defined on the manifold of symmetric positive definite matrices $\mathbb{S}_{++}^n$.

* $g_1(X) = \dfrac{1}{12} \left( \log(\det(X)) \right)^4$

* $g_2(X) = \left( \log(\det(X)) \right)^2$

* $h(X) = \log(\det(X))$

Required Julia Packages
The adap-RPPM algorithm requires the following Julia packages:

* [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/)

* [ManifoldDiff.jl](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/)

* [Manopt.jl](https://manoptjl.org/stable/)

* [LinearAlgebra.jl](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (standard library)

