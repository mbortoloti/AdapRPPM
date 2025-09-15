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

NAME_STRING = "EX02_"

# Set dimensions and number of initial guesses (for each dimension) for analysis
n_0 = parse(Int64,ARGS[1])
n_f = parse(Int64,ARGS[2])
Δn = parse(Int64,ARGS[3])
nguess = parse(Int64,ARGS[4])

NAME_STRING = NAME_STRING * ARGS[5]

# Echo file
file = open(NAME_STRING * "_ECHO.dat","w")

#################################################################################################
# Warmup
# (only for loading required lybrary) 
#
#################################################################################################

wg1(M,X) = la.logdet(X)^4
wgrad_g1(M,X) = 4.0*la.logdet(X)^3 * inv(X)

wg2(M,X) = 0.0
wgrad_g2(M,X) = zeros(2,2)

wh(M,X) = la.logdet(X)^2
w∂h(M,X) = 2.0*la.logdet(X) * inv(X)
#w∂H(M,X) = w∂h(X)

wg(M,X) = wg1(M,X) + wg2(M,X)
wf(M,X) = wg(M,X) - wh(M,X)
wgrad_g(M,X) = wgrad_g1(M,X) + wgrad_g2(M,X)


M = mf.SymmetricPositiveDefinite(2)
X0 = rand(M)
#λk(n) = sqrt(n)
adap_rppm(M,X0,wg1,wgrad_g1,wg2,wgrad_g2,wh,w∂h,1.e-2,5,1.e-4); 

Manopt.difference_of_convex_algorithm(M,wf,wg,w∂h,X0;grad_g=wgrad_g,
         #debug = [:Iteration, :Cost, :Change, " | ", 25, "\n", :Stop],
         stopping_criterion=StopAfterIteration(3)|StopWhenChangeLess(M,1.e-3),
         sub_stopping_criterion=StopAfterIteration(3)|StopWhenGradientNormLess(1.e-3)
       );

λ(k) = 1/(2.0*k)
Manopt.difference_of_convex_proximal_point(M,w∂h,X0;g=wg,grad_g=wgrad_g,λ=λ,
           #debug = [:Iteration, :Change, " | ", 25, "\n", :Stop],
           stopping_criterion=StopAfterIteration(3)|StopWhenChangeLess(M,1.e-3),
           sub_stopping_criterion=StopAfterIteration(3)|StopWhenGradientNormLess(1.e-3)
        );
##################################################################################################


dim = [i for i in n_0:Δn:n_f];

seed = MersenneTwister(1234)

global T = Matrix{Float64}(undef,size(dim,1)*nguess,3)
global ntest = 0


for n in dim

g1(M,X) = la.logdet(X)^4
grad_g1(M,X) = 4.0*la.logdet(X)^3 * inv(X)

g2(M,X) = 0.0
grad_g2(M,X) = zeros(n,n)

h(M,X) = la.logdet(X)^2
∂h(M,X) = 2.0*la.logdet(X) * inv(X)


g(M,X) = g1(M,X) + g2(M,X)
f(M,X) = g(M,X) - h(M,X)
grad_g(M,X) = grad_g1(M,X) + grad_g2(M,X)
#∂H(M,X) = ∂h(X)

#f(X) = g1(M,X) + g2(M,X) - h(M,X)

for _ in 1:nguess
       
    global ntest += 1
    
    # Set Manifold
    global M = mf.SymmetricPositiveDefinite(n)

    # Set initial guess
    global X0 = mf.rand(seed,M)
    #global X0 = log(n)*Matrix{Float64}(la.I,n,n)
    #maxiter = 1000
    ϵ = 1.e-8
    maxiter = 1000
    ########################################################################
    #                      PPMNN method analysis
    ########################################################################
    #println("PPMNN Analysis n = $n")
    #println("n=$n")
    try
        λ0 = 1.e-4
        local t0 = time();
        S,error,iter,λk = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ);                
        # S,error = ppmnn_1(M,g1,grad_g1,g2,grad_g2,h,grad_h,X0)
        local et = time() - t0;
        T[ntest,3] = et
        @printf(     "adap-RPPM ::%5d %15.10e %15.10e %+15.10e %8.5e %5d\n",n,et,la.det(S),f(M,S),λk,iter);
        @printf(file,"adap-RPPM ::%5d %15.10e %15.10e %+15.10e %8.5e %5d\n",n,et,la.det(S),f(M,S),λk,iter);
        #println("$S")
    catch error
        T[ntest,3] = Inf
        @printf(     "adap-RPPM ::%5d %15.10e\n",n,T[ntest,3]);
        @printf(file,"adap-RPPM ::%5d %15.10e\n",n,T[ntest,3]);
        println("$error")
    end
        
    ########################################################################
    #                      End PPMNN analysis
    ########################################################################

    ########################################################################
    #      Difference of Convex Method analysis
    ########################################################################
    try
     local t0 = time()
      S = Manopt.difference_of_convex_algorithm(M,f,g,∂h,X0;grad_g=grad_g,
         #debug = [:Iteration, :Cost, :Change, " | ", 25, "\n", :Stop],
         stopping_criterion=StopAfterIteration(maxiter)|StopWhenChangeLess(M,ϵ),
         sub_stopping_criterion=StopAfterIteration(5000)|StopWhenGradientNormLess(1.e-10)
       );
      local et = time() - t0
      T[ntest,2] = et
      @printf("DCA       ::%5d %15.10e %15.10e %+15.10e\n",n,et,la.det(S),f(M,S));
      @printf(file,"DCA       ::%5d %15.10e %15.10e %+15.10e\n",n,et,la.det(S),f(M,S));
    catch
        T[ntest,2] = Inf
        @printf("DCA       ::%5d %15.10e\n",n,T[ntest,2]);
        @printf(file,"DCA       ::%5d %15.10e\n",n,T[ntest,2]);
    end
        
    #######################################################################
    #     End of DC method analysis
    #######################################################################

    #######################################################################
    #               DCPP Method Analysis
    #######################################################################
    try
        λ(_) = 1/(2.0*n)
        local t0 = time()
        S = Manopt.difference_of_convex_proximal_point(M,∂h,X0;g=g,grad_g=grad_g,λ=λ,
           #debug = [:Iteration, :Change, " | ", 25, "\n", :Stop],
           #stopping_criterion=StopAfterIteration(maxiter)|StopWhenGradientNormLess(ϵ),
           stopping_criterion=StopAfterIteration(maxiter)|StopWhenChangeLess(M,ϵ),
           sub_stopping_criterion=StopAfterIteration(5000)|StopWhenGradientNormLess(1.e-10)
        );
        local et = time() - t0
        T[ntest,1] = et
        @printf("DCPPA     ::%5d %15.10e %15.10e %+15.10e\n",n,et,la.det(S),f(M,S));
        @printf(file,"DCPPA     ::%5d %15.10e %15.10e %+15.10e\n",n,et,la.det(S),f(M,S));
    catch
            T[ntest,1] = Inf
            @printf("DCPPA    ::%5d %15.10e\n",n,T[ntest,1]);
            @printf(file,"DCPPA    ::%5d %15.10e\n",n,T[ntest,1]);
    end
    #######################################################################
    #    End of DCPP Analysis
    #######################################################################
end    
end

close(file)

# Generate and write plot of performance profiles
ENV["PLOTS_TEST"]="true"
ENV["GKSwstype"] = "100"
performance_profile(PlotsBackend(),T,["DCPPA","DCA","Adap-RPPM"],leg=:bottomright,l=2,ylabel="Solved Problems (%)",xlabel="Performance ratio: CPU time")
savefig(NAME_STRING * "ADAP_RPPM" * "PP")

# Generate and write plot of n x time 
mdim = dim .* ( dim .+ 1.0) ./ 2.0
plot(mdim,T[:,1],label="DCPPA",lw=2,xticks = ([0.0,50.0,100.0,150.0,200.0],["0","50","100","150","200"]))
plot!(mdim,T[:,2],label="DCA",lw=2)
plot!(mdim,T[:,3],label="Adap-RPPM",lw=2)
xlabel!("Manifold Dimension "*L"d = " * "dim " * L"\mathbb{P}^n_{++}")
ylabel!("run time (sec.)")

if nguess == 1
    savefig(NAME_STRING * "DvT" * "Adap-RPPM")
end

# Writing Time for Performance Profile
writedlm(NAME_STRING * "time_PP.csv",T,',')
