using Test
using TensorInference
using Random

@testset "TensorNetworkModel file I/O" begin
    # Create a test model
    n = 3
    chi = 2
    d = 2
    tn = random_matrix_product_state(n, chi, d)
    
    # Add evidence for testing
    tn.evidence[1] = 0
    tn.evidence[2] = 1
    
    # Create temporary directory
    test_dir = mktempdir()
    
    # Test saving
    @testset "Saving" begin
        save_tensor_network(tn; folder=test_dir)
        @test isfile(joinpath(test_dir, "code.json"))
        @test isfile(joinpath(test_dir, "tensors.json"))
        @test isfile(joinpath(test_dir, "model.json"))
    end
    
    # Test loading
    @testset "Loading" begin
        tn_loaded = load_tensor_network(test_dir)
        
        # Verify basic properties
        @test tn_loaded.nvars == tn.nvars
        @test tn_loaded.evidence == tn.evidence
        @test tn_loaded.unity_tensors_idx == tn.unity_tensors_idx
        
        # Verify code structure
        @test tn_loaded.code isa typeof(tn.code)
        
        # Verify tensors
        @test length(tn_loaded.tensors) == length(tn.tensors)
        for (t_orig, t_loaded) in zip(tn.tensors, tn_loaded.tensors)
            @test size(t_orig) == size(t_loaded)
            @test eltype(t_orig) == eltype(t_loaded)
            @test Array(t_orig) ≈ Array(t_loaded)
        end
        
        # Verify model functionality
        @test probability(tn)[] ≈ probability(tn_loaded)[]
    end
    
    # Test error handling
    @testset "Error handling" begin
        # Invalid directory
        @test_throws SystemError load_tensor_network("nonexistent_directory")
        
        # Missing files
        for file in ["code.json", "tensors.json", "model.json"]
            bad_dir = mktempdir()
            save_tensor_network(tn; folder=bad_dir)
            rm(joinpath(bad_dir, file))
            @test_throws SystemError load_tensor_network(bad_dir)
            rm(bad_dir, recursive=true)
        end
    end
    
    # Clean up
    rm(test_dir, recursive=true)
end

@testset "Tensor serialization" begin
    Random.seed!(42)
    
    # Test real tensor
    real_tensor = rand(2, 2)
    dict_real = TensorInference.tensor_to_dict(real_tensor)
    @test TensorInference.tensor_from_dict(dict_real) ≈ real_tensor
    
    # Test complex tensor
    complex_tensor = rand(ComplexF64, 2, 2)
    dict_complex = TensorInference.tensor_to_dict(complex_tensor)
    @test TensorInference.tensor_from_dict(dict_complex) ≈ complex_tensor
    
    # Test higher-dimensional tensor
    high_dim_tensor = rand(2, 3, 4)
    dict_high_dim = TensorInference.tensor_to_dict(high_dim_tensor)
    @test TensorInference.tensor_from_dict(dict_high_dim) ≈ high_dim_tensor
    
    # Test invalid input
    @test_throws KeyError TensorInference.tensor_from_dict(Dict())
    @test_throws KeyError TensorInference.tensor_from_dict(Dict("size" => [2,2]))
end 