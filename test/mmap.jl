using Test
using Random
using OMEinsum
using TensorInference
using TensorInference: Factor, UAIModel, get_vars

@testset "clustering" begin
    ixs = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]
    @test TensorInference.connected_clusters(ixs, [2, 3, 6]) == [[2, 3] => [1, 2], [6] => [3]]
end

@testset "gradient-based tensor network solvers" begin
    problem = problem_from_artifact("uai2014", "MAR", "Promedus", 14)
    model, evidence = read_model(problem), read_evidence(problem)

    optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
    tn_ref = TensorNetworkModel(model; optimizer, evidence)

    # Does not marginalize any var
    mmap = MMAPModel(model; optimizer, queryvars=collect(1:model.nvars), evidence)
    @debug(mmap)
    @test maximum_logp(tn_ref) ≈ maximum_logp(mmap)

    # Marginalize all vars
    mmap2 = MMAPModel(model; optimizer, queryvars=Int[], evidence)
    @debug(mmap2)
    @test Array(probability(tn_ref))[] ≈ exp(maximum_logp(mmap2)[])

    # Does not optimize over open vertices
    mmap3 = MMAPModel(model; optimizer, queryvars=setdiff(1:model.nvars, [2, 4, 6]), evidence)
    @debug(mmap3)
    logp, config = most_probable_config(mmap3)
    @test log_probability(mmap3, config) ≈ logp
end

@testset "MMAP exhaustive small independent oracle" begin
    rng = MersenneTwister(4279)
    for trial in 1:12
        f = trial == 1 ? [1.0 0.0; 0.0 1.0] : rand(rng, 2, 2)
        g = trial == 1 ? [0.0 1.0; 1.0 0.0] : rand(rng, 2, 2)
        model = UAIModel(3, [2, 2, 2], [Factor((1, 2), f), Factor((2, 3), g)])
        for queryvars in ([1], [1, 3]), evidence in (Dict{Int,Int}(), Dict(3 => 1))
            mmap = MMAPModel(model; queryvars, evidence)
            scores = Dict{Tuple,Float64}()
            expected_outputvars = sort!(union(queryvars, collect(keys(evidence))))
            @test get_vars(mmap) == expected_outputvars
            for x in 0:1, y in 0:1, z in 0:1
                assignment = [x, y, z]
                all(assignment[k] == value for (k, value) in evidence) || continue
                key = Tuple(assignment[k] for k in expected_outputvars)
                scores[key] = get(scores, key, 0.0) + f[x + 1, y + 1] * g[y + 1, z + 1]
            end
            logp, config = most_probable_config(mmap)
            @test exp(logp) ≈ maximum(values(scores))
            @test scores[Tuple(config)] ≈ maximum(values(scores))
            @test log_probability(mmap, config) ≈ logp
        end
    end
end

@testset "UAI feasible reference lower bounds" begin
    problem_sets = dataset_from_artifact("uai2014")["MMAP"]
    problems = [
        ("Segmentation", 12, TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)),
        ("Segmentation", 13, TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)),
        ("Segmentation", 14, TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40))
    ]
    for (problem_set_name, id, optimizer) in problems
        @testset "$(problem_set_name) problem set, id = $id" begin
            problem = problem_sets[problem_set_name][id]
            @info "Testing: $(problem_set_name)_$id"
            model = MMAPModel(read_model(problem); optimizer, evidence=read_evidence(problem), queryvars=read_queryvars(problem))
            logp, solution = most_probable_config(model)
            reference_logp = log_probability(model, read_solution(problem))

            # These external assignments are feasible lower bounds, not certified
            # optima. A different assignment is valid when its objective is at
            # least as good; the small tests above independently establish that
            # the solver finds exact MMAP optima on exhaustively enumerable models.
            @test log_probability(model, solution) ≈ logp
            @test logp >= reference_logp || isapprox(logp, reference_logp)
        end
    end
end
