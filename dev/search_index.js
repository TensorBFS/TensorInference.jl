var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TensorInference","category":"page"},{"location":"#TensorInference","page":"Home","title":"TensorInference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for TensorInference.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TensorInference]","category":"page"},{"location":"#TensorInference.Factor","page":"Home","title":"TensorInference.Factor","text":"struct Factor{T, N}\n\nFields\n\nvars\nvals\n\nEncodes a discrete function over the set of variables vars that maps each instantiation of vars into a nonnegative number in vals.\n\n\n\n\n\n","category":"type"},{"location":"#TensorInference.TensorNetworkModeling","page":"Home","title":"TensorInference.TensorNetworkModeling","text":"struct TensorNetworkModeling{LT, ET, MT<:AbstractArray}\n\nProbabilistic modeling with a tensor network.\n\nFields\n\nvars is the degree of freedoms in the tensor network.\ncode is the tensor network contraction pattern.\ntensors is the tensors fed into the tensor network.\nfixedvertices is a dictionary to specifiy degree of freedoms fixed to certain values.\n\n\n\n\n\n","category":"type"},{"location":"#TensorInference.TensorNetworkModeling-Union{Tuple{LT}, Tuple{AbstractVector{LT}, OMEinsum.EinCode, Vector{<:AbstractArray}}} where LT","page":"Home","title":"TensorInference.TensorNetworkModeling","text":"TensorNetworkModeling(\n    vars::AbstractArray{LT, 1},\n    rawcode::OMEinsum.EinCode,\n    tensors::Vector{<:AbstractArray};\n    fixedvertices,\n    optimizer,\n    simplifier\n) -> TensorNetworkModeling\n\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.TensorNetworkModeling-Union{Tuple{LT}, Tuple{T}, Tuple{AbstractVector{LT}, Vector{<:TensorInference.Factor{T}}}} where {T, LT}","page":"Home","title":"TensorInference.TensorNetworkModeling","text":"TensorNetworkModeling(\n    vars::AbstractArray{LT, 1},\n    factors::Array{<:TensorInference.Factor{T}, 1};\n    openvertices,\n    fixedvertices,\n    optimizer,\n    simplifier\n) -> TensorNetworkModeling\n\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.backward_tropical-Tuple{Any, Tuple, Any, Any, Any, Any}","page":"Home","title":"TensorInference.backward_tropical","text":"backward_tropical(\n    ixs,\n    xs::Tuple,\n    iy,\n    y,\n    ymask,\n    size_dict\n) -> Vector{Any}\n\n\nThe backward rule for tropical einsum.\n\nixs and xs are labels and tensor data for input tensors,\niy and y are labels and tensor data for the output tensor,\nymask is the boolean mask for gradients,\nsize_dict is a key-value map from tensor label to dimension size.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.get_vars-Tuple{TensorNetworkModeling}","page":"Home","title":"TensorInference.get_vars","text":"get_vars(tn::TensorNetworkModeling) -> Vector\n\n\nGet the variables in this tensor network, they is also known as legs, labels, or degree of freedoms.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.marginals-Tuple{TensorNetworkModeling}","page":"Home","title":"TensorInference.marginals","text":"marginals(tn::TensorNetworkModeling) -> Vector\n\n\nReturns the marginal probability distribution of variables. One can use get_vars(tn) to get the full list of variables in this tensor network.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.maximum_logp-Tuple{TensorNetworkModeling}","page":"Home","title":"TensorInference.maximum_logp","text":"maximum_logp(\n    tn::TensorNetworkModeling\n) -> AbstractArray{<:TropicalNumbers.Tropical}\n\n\nReturns an output array containing largest log-probabilities.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.most_probable_config-Tuple{TensorNetworkModeling}","page":"Home","title":"TensorInference.most_probable_config","text":"most_probable_config(\n    tn::TensorNetworkModeling\n) -> Tuple{TropicalNumbers.Tropical, Vector}\n\n\nReturns the largest log-probability and the most probable configuration.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.probability-Tuple{TensorNetworkModeling, Any}","page":"Home","title":"TensorInference.probability","text":"probability(tn::TensorNetworkModeling, config) -> Real\n\n\nEvaluate the probability of config.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.probability-Tuple{TensorNetworkModeling}","page":"Home","title":"TensorInference.probability","text":"probability(tn::TensorNetworkModeling) -> AbstractArray\n\n\nContract the tensor network and return a probability array with its rank specified in the contraction code tn.code. The returned array may not be l1-normalized even if the total probability is l1-normalized, because the evidence tn.fixedvertices may not be empty.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.read_td_file-Tuple{AbstractString}","page":"Home","title":"TensorInference.read_td_file","text":"read_td_file(\n    td_filepath::AbstractString\n) -> Tuple{Int64, Int64, Int64, Vector{Vector{Int64}}, Vector{Vector{Int64}}}\n\n\nParse a tree decomposition instance described the PACE format.\n\nThe PACE file format is defined in: https://pacechallenge.org/2017/treewidth/\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.read_uai_evid_file-Tuple{AbstractString}","page":"Home","title":"TensorInference.read_uai_evid_file","text":"read_uai_evid_file(\n    uai_evid_filepath::AbstractString\n) -> Tuple{Vector{Int64}, Vector{Int64}}\n\n\nReturn the observed variables and values in uai_evid_filepath. If the passed file path is an empty string, return empty vectors.\n\nThe UAI file formats are defined in: https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.read_uai_file-Tuple{Any}","page":"Home","title":"TensorInference.read_uai_file","text":"read_uai_file(\n    uai_filepath;\n    factor_eltype\n) -> Tuple{Int64, Vector{Int64}, Int64, Any}\n\n\nParse the problem instance found in uai_filepath defined in the UAI model format.\n\nThe UAI file formats are defined in: https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.read_uai_mar_file-Tuple{AbstractString}","page":"Home","title":"TensorInference.read_uai_mar_file","text":"read_uai_mar_file(\n    uai_mar_filepath::AbstractString;\n    factor_eltype\n) -> Vector{Vector{Float64}}\n\n\nReturn the marginals of all variables. The order of the variables is the same as in the model\n\nThe UAI file formats are defined in: https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html\n\n\n\n\n\n","category":"method"}]
}
