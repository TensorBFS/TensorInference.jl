module TensorInference

using OMEinsum, LinearAlgebra
using DocStringExtensions, TropicalNumbers
using Artifacts
# The Tropical GEMM support
using TropicalGEMM
using StatsBase

# reexport OMEinsum functions
export RescaledArray
export contraction_complexity, TreeSA, GreedyMethod, KaHyParBipartite, SABipartite, MergeGreedy, MergeVectors

# read and load uai files
export read_uai_file, read_td_file, read_uai_evid_file, read_uai_mar_file, read_uai_problem, uai_problem_from_file
export set_evidence!

# marginals
export TensorNetworkModel, get_vars, get_cards, log_probability, probability, marginals

# MAP
export most_probable_config, maximum_logp

# sampling
export sample

# MMAP
export MMAPModel

include("Core.jl")
include("RescaledArray.jl")
include("utils.jl")
include("inference.jl")
include("maxprob.jl")
include("mmap.jl")
include("sampling.jl")

using Requires
function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

import PrecompileTools
PrecompileTools.@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    #PrecompileTools.@compile_workload begin
        #include("../example/asia/asia.jl")
    #end
end

end # module
