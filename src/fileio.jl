"""
    save_tensor_network(tn::TensorNetworkModel; folder::String)

Save a tensor network model to a folder with separate files for code, tensors, and model metadata.
The code is saved using `OMEinsum.writejson`, tensors as JSON, and model specifics in model.json.

# Arguments
- `tn::TensorNetworkModel`: The tensor network model to save
- `folder::String`: The folder path to save the files

# Files Created
- `code.json`: Contains the einsum code using OMEinsum format
- `tensors.json`: Contains the tensor data as JSON
- `model.json`: Contains nvars, evidence, and unity_tensors_idx

# Example
```julia
tn = TensorNetworkModel(...)  # create your model
save_tensor_network(tn; folder="my_model")
```
"""
function save_tensor_network(tn::TensorNetworkModel; folder::String)
    !isdir(folder) && mkpath(folder)

    # save code
    OMEinsum.writejson(joinpath(folder, "code.json"), tn.code)
    
    # save tensors
    open(joinpath(folder, "tensors.json"), "w") do io
        JSON.print(io, [tensor_to_dict(tensor) for tensor in tn.tensors], 2)
    end

    # save model metadata
    open(joinpath(folder, "model.json"), "w") do io
        JSON.print(io, Dict(
            "nvars" => tn.nvars,
            "evidence" => tn.evidence,
            "unity_tensors_idx" => tn.unity_tensors_idx
        ), 2)
    end
    return nothing
end

"""
    load_tensor_network(folder::String)

Load a tensor network model from a folder containing code, tensors, and model files.

# Arguments
- `folder::String`: The folder path containing the files

# Returns
- `TensorNetworkModel`: The loaded tensor network model

# Required Files
- `code.json`: Contains the einsum code using OMEinsum format
- `tensors.json`: Contains the tensor data as JSON
- `model.json`: Contains nvars, evidence, and unity_tensors_idx

# Example
```julia
tn = load_tensor_network("my_model")
```
"""
function load_tensor_network(folder::String)::TensorNetworkModel
    !isdir(folder) && throw(SystemError("Folder not found: $folder"))
    
    code_path = joinpath(folder, "code.json")
    tensors_path = joinpath(folder, "tensors.json")
    model_path = joinpath(folder, "model.json")
    !isfile(code_path) && throw(SystemError("Code file not found: $code_path"))
    !isfile(tensors_path) && throw(SystemError("Tensors file not found: $tensors_path"))
    !isfile(model_path) && throw(SystemError("Model file not found: $model_path"))
    
    code = OMEinsum.readjson(code_path)
    
    tensors = [tensor_from_dict(t) for t in JSON.parsefile(tensors_path)]

    model_dict = JSON.parsefile(model_path)
    
    # Convert evidence keys to Int (JSON parses them as strings)
    evidence = Dict{Int, Int}()
    for (k, v) in model_dict["evidence"]
        evidence[parse(Int, k)] = v
    end
    
    return TensorNetworkModel(
        model_dict["nvars"],
        code,
        tensors,
        evidence,
        collect(Int, model_dict["unity_tensors_idx"])
    )
end

"""
    tensor_to_dict(tensor::AbstractArray{T}) where T

Convert a tensor to a dictionary representation for JSON serialization.

# Arguments
- `tensor::AbstractArray{T}`: The tensor to convert

# Returns
- `Dict`: A dictionary containing tensor metadata and data

# Dictionary Structure
- `"size"`: The dimensions of the tensor
- `"complex"`: Boolean indicating if the tensor contains complex numbers
- `"data"`: The tensor data as a flat array of real numbers
"""
function tensor_to_dict(tensor::AbstractArray{T}) where T
    d = Dict()
    d["size"] = collect(size(tensor))
    d["complex"] = T <: Complex
    d["data"] = vec(reinterpret(real(T), tensor))
    return d
end

"""
    tensor_from_dict(dict::Dict)

Convert a dictionary back to a tensor.

# Arguments
- `dict::Dict`: The dictionary representation of a tensor

# Returns
- `AbstractArray`: The reconstructed tensor

# Dictionary Structure Expected
- `"size"`: The dimensions of the tensor
- `"complex"`: Boolean indicating if the tensor contains complex numbers
- `"data"`: The tensor data as a flat array of real numbers
"""
function tensor_from_dict(dict::Dict)
    size_vec = Tuple(dict["size"])
    is_complex = dict["complex"]
    data = collect(Float64, dict["data"])
    
    if is_complex
        complex_data = reinterpret(ComplexF64, data)
        return reshape(complex_data, size_vec...)
    else
        return reshape(data, size_vec...)
    end
end 