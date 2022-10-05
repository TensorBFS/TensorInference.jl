using Test
using OMEinsum
using TensorInference

@testset "gradient based tensor network solvers" begin
    ################# Load problem ####################
    instance = read_uai_problem("Promedus_14")

    # does not optimize over open vertices
    tn = TensorNetworkModeling(instance; optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40))
    @info(tn)
    @info timespace_complexity(tn)
    @time marginals2 = marginals(tn)
    # for dangling vertices, the output size is 1.
    npass = 0
    for i=1:instance.nvars
        npass += (length(marginals2[i]) == 1 && instance.reference_marginals[i] == [0.0, 1]) || isapprox(marginals2[i], instance.reference_marginals[i]; atol=1e-6)
    end
    @test npass == instance.nvars
end 
