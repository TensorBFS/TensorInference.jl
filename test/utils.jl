using TensorInference, Test

@testset "tensor train" begin
    tt = random_tensor_train_uai(Float64, 5, 3)
    @test tt.nvars == length(unique(vcat([collect(Int, f.vars) for f in tt.factors]...)))

    tt = random_tensor_train_uai(Float64, 5, 3; periodic=true)
    @test tt.nvars == length(unique(vcat([collect(Int, f.vars) for f in tt.factors]...)))
end

@testset "mps" begin
    tt = random_matrix_product_uai(Float64, 5, 3)
    @test tt.nvars == length(unique(vcat([collect(Int, f.vars) for f in tt.factors]...)))
end

