import Manifolds as mf
import ManifoldDiff as md
using Manopt
using LinearAlgebra

function build_fk(M :: mf.AbstractManifold, Y :: Matrix{Float64}, Zk :: Matrix{Float64}, 
        g1 :: Function, λk :: Float64)

    return g1(M,Y) + 0.5 * λk * mf.distance(M,Y,Zk)^2

end
###########################################################################################################
function build_grad_fk(M :: mf.AbstractManifold, Y :: Matrix{Float64}, Zk :: Matrix{Float64}, 
        grad_g1 :: Function, λk :: Float64)

   # r_grad_g1_Y = Y * grad_g1(M,Y) * Y
    r_grad_g1_Y = md.riemannian_gradient(M,Y,grad_g1(M,Y))

    return r_grad_g1_Y - λk * mf.log(M,Y,Zk)

end
###########################################################################################################
function adap_rppm(M :: mf.AbstractManifold, X0 :: Matrix{Float64}, g1 :: Function, grad_g1 :: Function, 
        g2 :: Function, grad_g2 :: Function, h :: Function, grad_h :: Function, λ0 :: Float64,
        maxiter :: Int,ϵ :: Float64;return_state = false)

    # Set objective function
    f(M,X) = g1(M,X) + g2(M,X) - h(M,X)
    
    # Set initial guess
    Xk = copy(M,X0)
    println(0,"  ",f(M,Xk))
    f_Xk = f(M,Xk)
  
    # Set λk
    λk = λ0
        
    cost = []
    push!(cost,f_Xk)

    for iter in 1:maxiter

        # Calculate riemannian gradient of g2
        Vk = md.riemannian_gradient(M,Xk,grad_g2(M,Xk))

        # Calculate riemannian subgradient of h
        Wk = md.riemannian_gradient(M,Xk,grad_h(M,Xk))
        
        WkmVk = Wk-Vk 

        while true
            
            Arg_exp = WkmVk / λk
            Zk = mf.exp(M,Xk,Arg_exp)

            # Build functions for subproblem
            fk(M,Y) = build_fk(M,Y,Zk,g1,λk)
            grad_fk(M,Y) = build_grad_fk(M,Y,Zk,grad_g1,λk)

            try
                # Solving subproblem
                Yk = Manopt.trust_regions(M, fk, grad_fk, Zk;
                     stepsize=ArmijoLinesearch(M;contraction_factor=0.5,initial_stepsize=1.0,
                     stop_when_stepsize_less=1.e-16),
                     #debug=[:Iteration, :Cost, "\n",:Stop],
                     stopping_criterion=StopAfterIteration(5000)|StopWhenChangeLess(M,1e-12)
                     );
               
                global dist_Xk_Yk = mf.distance(M,Xk,Yk)

                f_Yk = f(M,Yk)


             if dist_Xk_Yk < ϵ
                    
                    println(iter,"  ",dist_Xk_Yk,"   ",f_Yk,"   ",λk)
                    
                    if return_state
                        return cost
                    else
                        return Yk,0,iter,λk
                    end
                end

                if ~(f_Yk - f_Xk > -0.25 * λk * dist_Xk_Yk^2)
                    Xk = Yk
                    f_Xk = f_Yk
                    break
                else
                    λk = 2.0 * λk
                end
             #end
          catch
            λk = 2.0 * λk
          end

        end

        push!(cost,f_Xk)

        # Print info
        println(iter,"  ",dist_Xk_Yk,"   ",f_Xk,"   ",λk)
        
        # Update sequence
        #Xk = Yk

    end
    
    return -1,-1,-1,-1

    
end
