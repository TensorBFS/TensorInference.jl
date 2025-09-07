using Test
using TensorInference
using OMEinsum

@testset "RescaledArray" begin
    # Test basic construction
    @testset "Construction" begin
        α = 2.0
        T = [1.0 2.0; 3.0 4.0]
        r = RescaledArray(α, T)
        
        @test r.log_factor == α
        @test r.normalized_value == T
        @test size(r) == (2, 2)
        @test size(r, 1) == 2
        @test size(r, 2) == 2
    end
    
    # Test rescale_array function
    @testset "rescale_array" begin
        T = [1.0 2.0; 3.0 4.0]
        r = TensorInference.rescale_array(T)
        
        # Maximum absolute value should be 1 in normalized_value
        @test maximum(abs, r.normalized_value) ≈ 1.0
        
        # Original array should be recoverable
        @test Array(r) ≈ T
        
        # Test with zero array
        zero_T = zeros(2, 2)
        r_zero = TensorInference.rescale_array(zero_T)
        @test r_zero.log_factor == 0.0
        @test r_zero.normalized_value == zero_T
    end
    
    # Test Array conversion
    @testset "Array conversion" begin
        α = 1.5
        T = [0.5 1.0; 0.25 0.75]
        r = RescaledArray(α, T)
        
        expected = exp(α) * T
        @test Array(r) ≈ expected
    end
    
    # Test indexing
    @testset "Indexing" begin
        α = 0.5
        T = [1.0 2.0; 3.0 4.0]
        r = RescaledArray(α, T)
        
        @test r[1, 1] ≈ T[1, 1] * exp(α)
        @test r[2, 2] ≈ T[2, 2] * exp(α)
        @test r[1:2, 1] ≈ T[1:2, 1] * exp(α)
    end
    
    # Test copy
    @testset "Copy" begin
        α = 1.0
        T = [1.0 2.0; 3.0 4.0]
        r = RescaledArray(α, T)
        r_copy = copy(r)
        
        @test r_copy.log_factor == r.log_factor
        @test r_copy.normalized_value == r.normalized_value
        @test r_copy.normalized_value !== r.normalized_value  # Different objects
    end
    
    # Test selectdim
    @testset "selectdim" begin
        T = reshape(Float64.(1:8), 2, 2, 2)  # Convert to Float64 to match log factor type
        α = 0.5
        r = RescaledArray(α, T)
        
        r_slice = selectdim(r, 3, 1)
        @test r_slice.log_factor == α
        @test r_slice.normalized_value == selectdim(T, 3, 1)
    end
    
    # Test einsum operations
    @testset "Einsum operations" begin
        # Create two rescaled arrays
        α1, α2 = 1.0, 1.5
        T1 = [1.0 0.5; 0.25 1.0]
        T2 = [0.5 1.0; 1.0 0.5]
        
        r1 = RescaledArray(α1, T1)
        r2 = RescaledArray(α2, T2)
        
        # Test matrix multiplication via einsum
        code = ein"ij,jk->ik"
        result = einsum(code, (r1, r2))
        
        # Compare with regular array multiplication
        expected_array = Array(r1) * Array(r2)
        @test Array(result) ≈ expected_array
        
        # The log factor should be the sum of input log factors plus rescaling
        @test result isa RescaledArray
    end
    
    # Test fill! and conj
    @testset "fill! and conj" begin
        α = 0.5
        T = [1.0 2.0; 3.0 4.0]
        r = RescaledArray(α, copy(T))
        
        # Test fill!
        fill!(r, 2.0)
        expected_fill_value = 2.0 / exp(α)
        @test all(x -> x ≈ expected_fill_value, r.normalized_value)
        
        # Test conj with complex numbers
        α_complex = 1.0 + 0.5im
        T_complex = [1.0+1.0im 2.0+2.0im; 3.0+3.0im 4.0+4.0im]
        r_complex = RescaledArray(α_complex, T_complex)
        r_conj = conj(r_complex)
        
        @test r_conj.log_factor == conj(α_complex)
        @test r_conj.normalized_value == conj(T_complex)
    end
    
    # Test show methods
    @testset "Display" begin
        α = 1.0
        T = [1.0 2.0]
        r = RescaledArray(α, T)
        
        # Test that show methods don't error
        @test sprint(show, r) isa String
        @test sprint(show, "text/plain", r) isa String
    end

    # Test copyto!
    @testset "copyto!" begin
        α = 2.0
        T = [1.0 2.0; 3.0 4.0]
        r = RescaledArray(α, T)
        r_copy = similar(r)
        copyto!(r_copy, r)
        @test Array(r_copy) ≈ Array(r)

        α = 2.0
        T = [1.0 2.0; 3.0 4.0]
        r = RescaledArray(α, T)
        r_copy = similar(r)
        copyto!(selectdim(r_copy, 1, 1), selectdim(r, 1, 1))
        @test Array(r_copy)[1, :] ≈ Array(r)[1, :]
        @test !(Array(r_copy)[2, :] ≈ Array(r)[2, :])
    end
end
