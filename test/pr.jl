using Test
using OMEinsum
using KaHyPar
using TensorInference

@testset "UAI Reference Solution Comparison" begin
    problems = dataset_from_artifact("uai2014")["PR"]
    problem_sets = [
        #("Alchemy", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)), # fails
        #("CSP", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("DBN", KaHyParBipartite(sc_target = 25)),
        #("Grids", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)), # fails
        #("linkage", TreeSA(ntrials = 3, niters = 20, βs = 0.1:0.1:40)), # fails
        #("ObjectDetection", TreeSA(ntrials = 1, niters = 5, βs = 1:0.1:100)),
        ("Pedigree", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("Promedus", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("relational", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)), # fails
        ("Segmentation", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100))
    ]
    for (problem_set_name, optimizer) in problem_sets
        @testset "$(problem_set_name) problem set" begin
            for (id, problem) in problems[problem_set_name]
                @info "Testing: $(problem_set_name)_$id"
                tn = TensorNetworkModel(read_instance(problem); optimizer, evidence=read_evidence(problem))
                solution = probability(tn) |> first |> log10
                @test isapprox(solution, read_solution(problem); atol = 1e-3)
            end
        end
    end
end
