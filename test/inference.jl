using Test
using OMEinsum
using TensorInference
import LinearAlgebra: normalize!

@testset "gradient based tensor network solvers" begin
    ################# Load problem ####################
    problem_number = "14"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(dirname(@__DIR__), "data", problem_number)
    uai_filepath = joinpath(problem_dir, problem_filename * ".uai")
    uai_evid_filepath = joinpath(problem_dir, problem_filename * ".uai.evid")
    uai_mar_filepath = joinpath(problem_dir, problem_filename * ".uai.MAR")
    td_filepath = joinpath(problem_dir, problem_filename * ".td")

    reference_marginals = read_uai_mar_file(uai_mar_filepath)
    obsvars, obsvals = read_uai_evid_file(uai_evid_filepath)
    nvars, cards, nclique, factors = read_uai_file(uai_filepath; factor_eltype=Float32)

    # does not optimize over open vertices
    rawcode = EinCode([[[i] for i in 1:nvars]..., [[factor.vars...] for factor in factors]...], Int[])  # labels for edge tensors
    tensors = [[ones(Float32, 2) for i=1:length(cards)]..., getfield.(factors, :vals)...]
    tn = TensorNetworksSolver(rawcode, tensors; fixedvertices=Dict(zip(obsvars, obsvals .- 1)), optimizer=TreeSA(ntrials=1))
    @info timespace_complexity(tn.code, OMEinsum.get_size_dict(getixsv(tn.code), tensors))
    marginals2 = normalize!.(TensorInference.gradient(tn.code, generate_tensors(tn))[1:length(cards)], 1)
    @time marginals2 = normalize!.(TensorInference.gradient(tn.code, generate_tensors(tn))[1:length(cards)], 1)
    # for dangling vertices, the output size is 1.
    npass = 0
    for i=1:nvars
        npass += (length(marginals2[i]) == 1 && reference_marginals[i] == [0.0, 1]) || isapprox(marginals2[i], reference_marginals[i])
    end
    @show npass, nvars
    @test npass == nvars
end 