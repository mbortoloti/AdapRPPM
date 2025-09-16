using LinearAlgebra
using Manifolds

include("adap_rppm.jl")


#include("ppmnn.jl")


n = 2; # dimension of the SPD matrices

ngueses = 1;

#seed = MersenneTwister(1234)

# g1 function 
g1(_,X) = logdet(X)^4 / 12.0
grad_g1(_,X) = logdet(X)^3 * inv(X) / 3.0

# g2 function
g2(_,X) = logdet(X)^2
grad_g2(_,X) = 2.0 * logdet(X) * inv(X)

# h function
h(M,X) = logdet(X)
∂h(M,X) = inv(X)
      
    
# Set Manifold
M = SymmetricPositiveDefinite(n)

#  Initial guess
X0 = log(n)*Matrix{Float64}(I,n,n)

# Tolerance for checking convergence
ϵ = 1.e-8

# Maximum number of iterations
maxiter = 1000

# Initial λ
λ0 = 1.e-4

S,error,iter,λk = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ);     

println("End of adap_rppm")           
   
