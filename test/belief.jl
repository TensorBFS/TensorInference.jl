using TensorInference, Test
using OMEinsum

@testset "process message" begin
    mi = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    mo_expected = [[6, 12, 20], [3, 8, 15], [2, 6, 12]]
    mo = similar.(mi)
    TensorInference._process_message!(mo, mi)
    @test mo == mo_expected
end

@testset "star code" begin
    code = TensorInference.star_code(3)
    c1, c2, c3, c4 = [DynamicNestedEinsum{Int}(i) for i in 1:4]
    ne1 = DynamicNestedEinsum([c1, c2], DynamicEinCode([[1, 2, 3], [1]], [2, 3]))
    ne2 = DynamicNestedEinsum([ne1, c3], DynamicEinCode([[2, 3], [2]], [3]))
    ne3 = DynamicNestedEinsum([ne2, c4], DynamicEinCode([[3], [3]], Int[]))
    @test code == ne3
    t = randn(2, 2, 2)
    v1 = randn(2)
    v2 = randn(2)
    v3 = randn(2)
    vectors_out = [similar(v1), similar(v2), similar(v3)]
    TensorInference._collect_message!(vectors_out, t, [v1, v2, v3])
    @test vectors_out[1] ≈ reshape(t, 2, 4) * kron(v3, v2)  # NOTE: v3 is the little end
    @test vectors_out[2] ≈ vec(v1' * reshape(reshape(t, 4, 2) * v3, 2, 2))
    @test vectors_out[3] ≈ vec(kron(v2, v1)' * reshape(t, 4, 2))
end

@testset "constructor" begin
    problem = problem_from_artifact("uai2014", "MAR", "Promedus", 14)
    uai = read_model(problem)
    bp = BeliefPropgation(uai)
    @test length(bp.v2t) == 414
    @test TensorInference.num_tensors(bp) == 414
    @test TensorInference.num_variables(bp) == length(unique(vcat([collect(Int, f.vars) for f in uai.factors]...)))
end

@testset "belief propagation" begin
    n = 5
    chi = 3
    mps_uai = TensorInference.random_tensor_train_uai(Float64, n, chi)
    bp = BeliefPropgation(mps_uai)
    @test TensorInference.initial_state(bp) isa TensorInference.BPState
    state, info = belief_propagate(bp)
    @test info.converged
    @test info.iterations < 10
    contraction_res = TensorInference.contraction_results(state)
    tnet = TensorNetworkModel(mps_uai)
    expected_result = probability(tnet)[]
    @test all(r -> isapprox(r, expected_result), contraction_res)
    mars = marginals(state)
    mars_tnet = marginals(tnet)
    for v in 1:TensorInference.num_variables(bp)
        @test mars[[v]] ≈ mars_tnet[[v]]
    end
end