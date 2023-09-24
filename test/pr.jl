using Test
using OMEinsum
using KaHyPar
using TensorInference

@testset "UAI Reference Solution Comparison" begin
    problems = dataset_from_artifact("uai2014")["PR"]
    problem_sets = [
        #("Alchemy", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("CSP", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("DBN", KaHyParBipartite(sc_target = 25)),
        #("Grids", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("linkage", TreeSA(ntrials = 3, niters = 20, βs = 0.1:0.1:40)),
        #("ObjectDetection", TreeSA(ntrials = 1, niters = 5, βs = 1:0.1:100)),
        ("Pedigree", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("Promedus", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("relational", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)), tw too large
        ("Segmentation", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100))
    ]
    for (problem_set_name, optimizer) in problem_sets
        @testset "$(problem_set_name) problem set" begin
            for (id, problem) in problems[problem_set_name]
                @info "Testing: $(problem_set_name)_$id"
                tn = TensorNetworkModel(read_model(problem); optimizer, evidence=read_evidence(problem))
                solution = log_probability(tn) / log(10) |> first
                @test isapprox(solution, read_solution(problem); atol = 1e-3, rtol=1e-3)
            end
        end
    end
end

@testset "issue 77" begin
    problems = dataset_from_artifact("uai2014")["PR"]
    problem_set_name = "Alchemy"
    optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
    id, problem = problems[problem_set_name] |> first
    tn = TensorNetworkModel(read_model(problem); optimizer, evidence=read_evidence(problem))
    solution = log_probability(tn) / log(10) |> first
    @test isapprox(solution, read_solution(problem); atol=1e-3)
end