import Manifolds as mf
import ManifoldDiff as md
using Manopt
import LinearAlgebra as la
using Printf
using Plots
using Random
using BenchmarkProfiles
using LaTeXStrings
using DelimitedFiles

include("adap_rppm.jl")
#include("ppmnn.jl")


n = 2; # dimension of the SPD matrices

ngueses = 1;

seed = MersenneTwister(1234)

    g1(_,X) = la.logdet(X)^4 / 12.0
    grad_g1(_,X) = la.logdet(X)^3 * inv(X) / 3.0


    g2(_,X) = la.logdet(X)^2
grad_g2(_,X) = 2.0 * la.logdet(X) * inv(X)

h(M,X) = la.logdet(X)
∂h(M,X) = la.inv(X)



       
    
    # Set Manifold
    global M = mf.SymmetricPositiveDefinite(n)

    # Set initial guess
    #global X0 = mf.rand(seed,M)
    global X0 = log(n)*Matrix{Float64}(la.I,n,n)
    #maxiter = 1000
    ϵ = 1.e-8
    maxiter = 1000
        λ0 = 1.e-4
        S,error,iter,λk = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ);     
        println("End of adap_rppm")           
        # S,error = ppmnn_1(M,g1,grad_g1,g2,grad_g2,h,grad_h,X0)
   
