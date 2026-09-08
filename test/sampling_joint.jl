using TensorInference, OMEinsum, Random, LinearAlgebra, Test

@testset "MPS sampling preserves the joint distribution" begin
    for T in (Float64, ComplexF64)
        Random.seed!(140)
        uai = random_matrix_product_uai(T, 4, 3)
        model = TensorNetworkModel(uai; optimizer=GreedyMethod())
        # Independently evaluate the four ket tensors by ordinary matrix products.
        # The remaining factors are their conjugates, so probabilities are |ψ|².
        ket = [factor.vals for factor in uai.factors[1:4]]
        weights = map(CartesianIndices((2, 2, 2, 2))) do index
            a, b, c, d = Tuple(index)
            amplitude = transpose(ket[1][a, :]) * ket[2][:, b, :] *
                        ket[3][:, c, :] * ket[4][:, d]
            abs2(amplitude)
        end
        probabilities = vec(weights) ./ sum(weights)
        n = 10000
        # Hoeffding + union bound: failure probability <= 10⁻⁸ across 16 bins.
        tolerance = sqrt(log(2length(probabilities) / 1e-8) / (2n))
        for batched in (false, true)
            Random.seed!(142)
            draws = batched ? sample(model, n; queryvars=collect(1:4)) :
                [copy(sample(model, 1; queryvars=collect(1:4))[1]) for _ in 1:n]
            counts = zeros(Int, length(probabilities))
            for draw in draws
                index = 1 + sum(draw[i] * 2^(i-1) for i in 1:4)
                counts[index] += 1
            end
            @test all(abs.(counts ./ n .- probabilities) .<= tolerance)
        end
    end
end
