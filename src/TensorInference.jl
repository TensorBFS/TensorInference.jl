"""
Main module for `TensorInference.jl` -- A toolbox for probabilistic inference using contraction of tensor networks.

# Exports

$(EXPORTS)
"""
module TensorInference

using OMEinsum, LinearAlgebra
using DocStringExtensions, TropicalNumbers
# The Tropical GEMM support
using StatsBase
using PrettyTables
import Pkg

# reexport OMEinsum functions
export RescaledArray
export contraction_complexity, TreeSA, GreedyMethod, KaHyParBipartite, SABipartite, MergeGreedy, MergeVectors

# read and load uai files
export read_model_file, read_td_file, read_evidence_file
export problem_from_artifact, ArtifactProblemSpec
export read_model, UAIModel, read_evidence, read_solution, read_queryvars, dataset_from_artifact

# marginals
export TensorNetworkModel, get_vars, get_cards, log_probability, probability, marginals, update_evidence!

# MAP
export most_probable_config, maximum_logp

# sampling
export sample

# MMAP
export MMAPModel

# utils
export random_matrix_product_state

include("Core.jl")
include("RescaledArray.jl")
include("utils.jl")
include("mar.jl")
include("map.jl")
include("mmap.jl")
include("sampling.jl")

using Requires
function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
    @require GenericTensorNetworks = "3521c873-ad32-4bb4-b63d-f4f178f42b49" include("generictensornetworks.jl")
end

# import PrecompileTools
# PrecompileTools.@setup_workload begin
#     # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
#     # precompile file and potentially make loading faster.
#     PrecompileTools.@compile_workload begin
#         include("../example/asia-network/main.jl")
#     end
# end

end # module
