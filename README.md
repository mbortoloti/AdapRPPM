

**An adaptative proximal point method for nonsmooth and nonconvex optimization on Hadamard manifolds**

Vitaliano Amaral, Marcio Bortoloti, Jurandir O. Lopes, Gilson Silva


**Abstract** This paper addresses a class of nonsmooth and nonconvex optimization problems defined on complete Riemannian manifolds. The objective function has a composite structure, combining convex, differentiable, and lower semicontinuous terms, thereby generalizing the classical framework of difference-of-convex programming. Motivated by recent advances in proximal point methods in both Euclidean and Riemannian settings, we propose two
variants of the proximal point method for solving this class of problems. The first variant requires prior knowledge of the Lipschitz constant of the gradient of the smooth part, making it suitable when this parameter can be readily computed. The second variant, in contrast, does not require such knowledge, thus broadening its applicability. We analyze the complexity of both approaches, establish their convergence, and illustrate their effectiveness through numerical experiments.

In order to run adap-RPPM, the functions and their euclidean gradients (or subgradients) must be provided.

Please, see the file "example.jl" for an example of running code, where is solved the following problem:

Minimize $$f(X)=g_1(X)+g_2(X)-h(X),$$ where $f$ is defined on $\mathbb{S}^{n}_{++}$, with $g_1(X)=\frac{1}{12} \log(\det(X))^4$, $g_2(X)=\log(\det(X))^2$ and $h(X) = \log(\det(X))$.

The adap_RPPM algorithm require the following Julia languages:

1. [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/)
1. [ManifoldDiff.jl](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/)
1. [Manopt.jl](https://manoptjl.org/stable/)
1. [LinearAlgebra.jl](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)