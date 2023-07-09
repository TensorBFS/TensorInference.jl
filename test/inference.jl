using Test, Artifacts
using OMEinsum
using KaHyPar
using TensorInference

@testset "composite number" begin
    A = RescaledArray(2.0, [2.0 3.0; 5.0 6.0])
    x = RescaledArray(2.0, [2.0, 3.0])
    op = ein"ij, j -> i"
    @test Array(x) ≈ exp(2.0) .* [2.0, 3.0]
    @test op(Array(A), Array(x)) ≈ Array(op(A, x))
end

@testset "cached, rescaled contract" begin
    problem = read_uai_problem("Promedus_14")
    ref_sol = problem.reference_marginals
    optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
    tn = TensorNetworkModel(problem; optimizer)
    p1 = probability(tn; usecuda = false, rescale = false)
    p2 = probability(tn; usecuda = false, rescale = true)
    @test p1 ≈ Array(p2)

    # cached contract
    xs = TensorInference.adapt_tensors(tn; usecuda = false, rescale = true)
    size_dict = OMEinsum.get_size_dict!(getixsv(tn.code), xs, Dict{Int, Int}())
    cache = TensorInference.cached_einsum(tn.code, xs, size_dict)
    @test cache.content isa RescaledArray
    @test Array(cache.content) ≈ p1

    # compute marginals
    ti_sol = marginals(tn)
    ref_sol[problem.obsvars] .= fill([1.0], length(problem.obsvars)) # imitate dummy vars
    @test isapprox(ti_sol, ref_sol; atol = 1e-5)
end

function get_problems(problem_set::String)
    # Capture the problem names that belong to the current problem_set
    regex = Regex("($(problem_set)_\\d*)(\\.uai)\$")
    return readdir(artifact"MAR_prob"; sort = false) |>
           x -> map(y -> match(regex, y), x) |> # apply regex
                x -> filter(!isnothing, x) |> # filter out `nothing` values
                     x -> map(first, x) # get the first capture of each element
end

@testset "gradient-based tensor network solvers" begin
    problem_sets = [
        #("Alchemy", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("CSP", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("DBN", KaHyParBipartite(sc_target = 25)),
        #("Grids", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)), # greedy also works
        #("linkage", TreeSA(ntrials = 3, niters = 20, βs = 0.1:0.1:40)), # linkage_15 fails
        #("ObjectDetection", TreeSA(ntrials = 1, niters = 5, βs = 1:0.1:100)),
        #("Pedigree", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)), # greedy also works
        ("Promedus", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)), # greedy also works
        #("relational", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),
        ("Segmentation", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100))  # greedy also works
    ]

    for (problem_set, optimizer) in problem_sets
        @testset "$(problem_set) problem_set" begin

            # Capture the problem names that belong to the current problem set
            problems = get_problems(problem_set)

            for problem in problems
                @info "Testing: $problem"
                @testset "$(problem)" begin
                    problem = read_uai_problem(problem)
                    ref_sol = problem.reference_marginals
                    obsvars = problem.obsvars

                    # does not optimize over open vertices
                    tn = TensorNetworkModel(problem; optimizer)
                    sc = contraction_complexity(tn).sc
                    sc > 28 && error("space complexity too large! got $(sc)")
                    @debug contraction_complexity(tn)
                    ti_sol = marginals(tn)
                    ref_sol[obsvars] .= fill([1.0], length(obsvars)) # imitate dummy vars
                    @test isapprox(ti_sol, ref_sol; atol = 1e-4)
                end
            end
        end
    end
end
