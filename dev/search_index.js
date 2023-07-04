var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TensorInference","category":"page"},{"location":"#TensorInference","page":"Home","title":"TensorInference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for TensorInference.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TensorInference]","category":"page"},{"location":"#TensorInference.Factor","page":"Home","title":"TensorInference.Factor","text":"struct Factor{T, N}\n\nFields\n\nvars\nvals\n\nEncodes a discrete function over the set of variables vars that maps each instantiation of vars into a nonnegative number in vals.\n\n\n\n\n\n","category":"type"},{"location":"#TensorInference.MMAPModel","page":"Home","title":"TensorInference.MMAPModel","text":"struct MMAPModel{LT, AT<:AbstractArray}\n\nComputing the most likely assignment to the query variables,  Xₘ ⊆ X after marginalizing out the remaining variables Xₛ = X \\ Xₘ.\n\nrm MMAP(X_iE=e) = arg max_X_M sum_X_S prod_F f(x_M x_S e)\n\nFields\n\nvars is the remaining (or not marginalized) degree of freedoms in the tensor network.\ncode is the tropical tensor network contraction pattern.\ntensors is the tensors fed into the tensor network.\nclusters is the clusters, each element of this cluster is a TensorNetworkModel instance for marginalizing certain variables.\nfixedvertices is a dictionary to specifiy degree of freedoms fixed to certain values, which should not have overlap with the marginalized variables.\n\n\n\n\n\n","category":"type"},{"location":"#TensorInference.MMAPModel-Tuple{TensorInference.UAIInstance}","page":"Home","title":"TensorInference.MMAPModel","text":"MMAPModel(instance::TensorInference.UAIInstance)\n\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.MMAPModel-Union{Tuple{LT}, Tuple{T}, Tuple{AbstractVector{LT}, Vector{<:TensorInference.Factor{T}}}} where {T, LT}","page":"Home","title":"TensorInference.MMAPModel","text":"MMAPModel(\n    vars::AbstractArray{LT, 1},\n    factors::Array{<:TensorInference.Factor{T}, 1}\n)\n\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.RescaledArray","page":"Home","title":"TensorInference.RescaledArray","text":"struct RescaledArray{T, N, AT<:AbstractArray{T, N}} <: AbstractArray{T, N}\n\nRescaledArray(α, T) -> RescaledArray\n\nAn array data type with a log-prefactor, and a l∞-normalized storage, i.e. the maximum element in a tensor is 1. This tensor type can avoid the potential underflow/overflow of numbers in a tensor network. The constructor RescaledArray(α, T) creates a rescaled array that equal to exp(α) * T.\n\n\n\n\n\n","category":"type"},{"location":"#TensorInference.TensorNetworkModel","page":"Home","title":"TensorInference.TensorNetworkModel","text":"struct TensorNetworkModel{LT, ET, MT<:AbstractArray}\n\nProbabilistic modeling with a tensor network.\n\nFields\n\nvars is the degree of freedoms in the tensor network.\ncode is the tensor network contraction pattern.\ntensors is the tensors fed into the tensor network.\nfixedvertices is a dictionary to specifiy degree of freedoms fixed to certain values.\n\n\n\n\n\n","category":"type"},{"location":"#TensorInference.TensorNetworkModel-Tuple{TensorInference.UAIInstance}","page":"Home","title":"TensorInference.TensorNetworkModel","text":"TensorNetworkModel(\n    instance::TensorInference.UAIInstance\n) -> TensorNetworkModel{Int64}\n\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.TensorNetworkModel-Union{Tuple{LT}, Tuple{AbstractVector{LT}, OMEinsum.EinCode, Vector{<:AbstractArray}}} where LT","page":"Home","title":"TensorInference.TensorNetworkModel","text":"TensorNetworkModel(\n    vars::AbstractArray{LT, 1},\n    rawcode::OMEinsum.EinCode,\n    tensors::Vector{<:AbstractArray}\n) -> TensorNetworkModel\n\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.TensorNetworkModel-Union{Tuple{LT}, Tuple{T}, Tuple{AbstractVector{LT}, AbstractVector{Int64}, Vector{<:TensorInference.Factor{T}}}} where {T, LT}","page":"Home","title":"TensorInference.TensorNetworkModel","text":"TensorNetworkModel(\n    vars::AbstractArray{LT, 1},\n    cards::AbstractVector{Int64},\n    factors::Array{<:TensorInference.Factor{T}, 1}\n) -> TensorNetworkModel\n\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.UAIInstance","page":"Home","title":"TensorInference.UAIInstance","text":"struct UAIInstance{ET, FT<:(TensorInference.Factor{ET})}\n\nFields\n\nnvars is the number of variables,\nnclique is the number of cliques,\ncards is a vector of cardinalities for variables,\nfactors is a vector of factors,\nobsvars is a vector of observed variables,\nobsvals is a vector of observed values,\nreference_marginals is a vector of marginal probabilities.\n\n\n\n\n\n","category":"type"},{"location":"#TensorInference.backward_tropical-Tuple{Any, Tuple, Vararg{Any, 4}}","page":"Home","title":"TensorInference.backward_tropical","text":"backward_tropical(\n    ixs,\n    xs::Tuple,\n    iy,\n    y,\n    ymask,\n    size_dict\n) -> Vector{Any}\n\n\nThe backward rule for tropical einsum.\n\nixs and xs are labels and tensor data for input tensors,\niy and y are labels and tensor data for the output tensor,\nymask is the boolean mask for gradients,\nsize_dict is a key-value map from tensor label to dimension size.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.get_cards-Tuple{MMAPModel}","page":"Home","title":"TensorInference.get_cards","text":"get_cards(mmap::MMAPModel) -> Vector\n\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.get_cards-Tuple{TensorNetworkModel}","page":"Home","title":"TensorInference.get_cards","text":"get_cards(tn::TensorNetworkModel) -> Vector\n\n\nGet the cardinalities of variables in this tensor network.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.get_vars-Tuple{MMAPModel}","page":"Home","title":"TensorInference.get_vars","text":"get_vars(mmap::MMAPModel) -> Vector\n\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.get_vars-Tuple{TensorNetworkModel}","page":"Home","title":"TensorInference.get_vars","text":"get_vars(tn::TensorNetworkModel) -> Vector\n\n\nGet the variables in this tensor network, they are also known as legs, labels, or degree of freedoms.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.log_probability-Tuple{TensorNetworkModel, Union{Dict, AbstractVector}}","page":"Home","title":"TensorInference.log_probability","text":"log_probability(\n    tn::TensorNetworkModel,\n    config::Union{Dict, AbstractVector}\n) -> Real\n\n\nEvaluate the log probability of config.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.marginals-Tuple{TensorNetworkModel}","page":"Home","title":"TensorInference.marginals","text":"marginals(tn::TensorNetworkModel) -> Vector\n\n\nReturns the marginal probability distribution of variables. One can use get_vars(tn) to get the full list of variables in this tensor network.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.maximum_logp-Tuple{TensorNetworkModel}","page":"Home","title":"TensorInference.maximum_logp","text":"maximum_logp(\n    tn::TensorNetworkModel\n) -> AbstractArray{<:Real}\n\n\nReturns an output array containing largest log-probabilities.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.most_probable_config-Tuple{TensorNetworkModel}","page":"Home","title":"TensorInference.most_probable_config","text":"most_probable_config(\n    tn::TensorNetworkModel\n) -> Tuple{Real, Vector}\n\n\nReturns the largest log-probability and the most probable configuration.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.probability-Tuple{TensorNetworkModel}","page":"Home","title":"TensorInference.probability","text":"probability(tn::TensorNetworkModel) -> AbstractArray\n\n\nContract the tensor network and return a probability array with its rank specified in the contraction code tn.code. The returned array may not be l1-normalized even if the total probability is l1-normalized, because the evidence tn.fixedvertices may not be empty.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.read_td_file-Tuple{AbstractString}","page":"Home","title":"TensorInference.read_td_file","text":"read_td_file(\n    td_filepath::AbstractString\n) -> Tuple{Int64, Int64, Int64, Vector{Vector{Int64}}, Vector{Vector{Int64}}}\n\n\nParse a tree decomposition instance described the PACE format.\n\nThe PACE file format is defined in: https://pacechallenge.org/2017/treewidth/\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.read_uai_evid_file-Tuple{AbstractString}","page":"Home","title":"TensorInference.read_uai_evid_file","text":"read_uai_evid_file(\n    uai_evid_filepath::AbstractString\n) -> Tuple{Vector{Int64}, Vector{Int64}}\n\n\nReturn the observed variables and values in uai_evid_filepath. If the passed file path is an empty string, return empty vectors.\n\nThe UAI file formats are defined in: https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.read_uai_file-Tuple{Any}","page":"Home","title":"TensorInference.read_uai_file","text":"read_uai_file(\n    uai_filepath\n) -> Tuple{Int64, Vector{Int64}, Int64, Any}\n\n\nParse the problem instance found in uai_filepath defined in the UAI model format.\n\nThe UAI file formats are defined in: https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.read_uai_mar_file-Tuple{AbstractString}","page":"Home","title":"TensorInference.read_uai_mar_file","text":"read_uai_mar_file(\n    uai_mar_filepath::AbstractString\n) -> Vector{Vector{Float64}}\n\n\nReturn the marginals of all variables. The order of the variables is the same as in the model\n\nThe UAI file formats are defined in: https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.read_uai_problem-Tuple{AbstractString}","page":"Home","title":"TensorInference.read_uai_problem","text":"read_uai_problem(\n    problem::AbstractString\n) -> TensorInference.UAIInstance{Float64, _A} where _A<:(TensorInference.Factor{Float64})\n\n\nRead a UAI problem from an artifact.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.rescale_array-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T","page":"Home","title":"TensorInference.rescale_array","text":"rescale_array(tensor::AbstractArray{T}) -> RescaledArray\n\n\nReturns a rescaled array that equivalent to the input tensor.\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.set_evidence!-Tuple{TensorInference.UAIInstance, Vararg{Pair{Int64}}}","page":"Home","title":"TensorInference.set_evidence!","text":"set_evidence!(\n    uai::TensorInference.UAIInstance,\n    pairs::Pair{Int64}...\n) -> TensorInference.UAIInstance\n\n\nSet the evidence of an UAI instance.\n\nExamples\n\njulia> problem = read_uai_problem(\"Promedus_14\"); problem.obsvars, problem.obsvals\n([42, 48, 27, 30, 29, 15, 124, 5, 148], [1, 1, 1, 1, 1, 1, 1, 1, 1])\n\njulia> set_evidence!(problem, 2=>0, 4=>1); problem.obsvars, problem.obsvals\n([2, 4], [0, 1])\n\n\n\n\n\n","category":"method"},{"location":"#TensorInference.uai_problem_from_file-Tuple{String}","page":"Home","title":"TensorInference.uai_problem_from_file","text":"uai_problem_from_file(\n    uai_filepath::String\n) -> TensorInference.UAIInstance{Float64, _A} where _A<:(TensorInference.Factor{Float64})\n\n\nRead a UAI problem from a file.\n\n\n\n\n\n","category":"method"}]
}
