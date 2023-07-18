using Test
using OMEinsum
using KaHyPar
using TensorInference

@testset "UAI Reference Solution Comparison" begin
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
            problem_names = TensorInference.get_problem_names("uai2014", problem_set_name, "PR")
            for problem_name in problem_names
                @testset "$(problem_name)" begin
                    @info "Testing: $problem_name"
                    instance = read_instance_from_artifact("uai2014", problem_name, "PR")
                    tn = TensorNetworkModel(instance; optimizer)
                    solution = probability(tn) |> first |> log10
                    @test isapprox(solution, instance.reference_solution; atol = 1e-3)
                end
            end
        end
    end
end
