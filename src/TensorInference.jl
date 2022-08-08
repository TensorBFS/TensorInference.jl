module TensorInference

using OMEinsum, LinearAlgebra
using DocStringExtensions, TropicalNumbers

# reexport OMEinsum functions
export timespace_complexity, timespacereadwrite_complexity, TreeSA, GreedyMethod, KaHyParBipartite, SABipartite, MergeGreedy, MergeVectors

# read and load uai files
export read_uai_file, read_td_file, read_uai_evid_file, read_uai_mar_file, read_uai_dir

# marginals
export TensorNetworkModeling, get_vars, get_cards, probability, marginals

# MAP
export most_probable_config, maximum_logp

# MMAP
export MMAPModeling

include("Core.jl")
include("utils.jl")
include("inference.jl")
include("maxprob.jl")
include("mmap.jl")

using Requires
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end # module
