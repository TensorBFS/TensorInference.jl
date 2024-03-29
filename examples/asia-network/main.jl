# # The ASIA network

# The graph below corresponds to the *ASIA network*, a simple Bayesian model
# used extensively in educational settings. It was introduced by Lauritzen in
# 1988 [^lauritzen1988local].

# ```
# ┌───┐           ┌───┐
# │ A │         ┌─┤ S ├─┐
# └─┬─┘         │ └───┘ │
#   │           │       │
#   ▼           ▼       ▼
# ┌───┐       ┌───┐   ┌───┐
# │ T │       │ L │   │ B │
# └─┬─┘       └─┬─┘   └─┬─┘
#   │   ┌───┐   │       │
#   └──►│ E │◄──┘       │
#       └─┬─┘           │
# ┌───┐   │   ┌───┐     │
# │ X │◄──┴──►│ D │◄────┘
# └───┘       └───┘
# ```

# The table below explains the meanings of each random variable used in the
# ASIA network model.

# | **Random variable**  | **Meaning**                     |
# |        :---:         | :---                            |
# |        ``A``         | Recent trip to Asia             |
# |        ``T``         | Patient has tuberculosis        |
# |        ``S``         | Patient is a smoker             |
# |        ``L``         | Patient has lung cancer         |
# |        ``B``         | Patient has bronchitis          |
# |        ``E``         | Patient hast ``T`` and/or ``L`` |
# |        ``X``         | Chest X-Ray is positive         |
# |        ``D``         | Patient has dyspnoea            |

# ---

# We now demonstrate how to use the TensorInference.jl package for conducting a
# variety of inference tasks on the Asia network.

# ---

# Import the TensorInference package, which provides the functionality needed
# for working with tensor networks and probabilistic graphical models.
using TensorInference

# ---

# Load the ASIA network model from the `asia.uai` file located in the examples
# directory. See [Model file format (.uai)](@ref) for a description of the
# format of this file.
model = read_model_file(pkgdir(TensorInference, "examples", "asia-network", "model.uai"))

# ---

# Create a tensor network representation of the loaded model.
tn = TensorNetworkModel(model)

# ---

# Calculate the partition function. Since the factors in this model are
# normalized, the partition function is the same as the total probability, $1$.
probability(tn) |> first

# ---

# Calculate the marginal probabilities of each random variable in the model.
marginals(tn)

# ---

# Retrieve all the variables in the model.
get_vars(tn)

# ---

# Set the evidence: Assume that the "X-ray" result (variable 7) is negative.
# Since setting the evidence may affect the contraction order of the tensor
# network, recompute it.
tn = TensorNetworkModel(model, evidence = Dict(7 => 0))

# ---

# Calculate the maximum log-probability among all configurations.
maximum_logp(tn)

# ---

# Generate 10 samples from the posterior distribution.
sample(tn, 10)

# ---

# Retrieve both the maximum log-probability and the most probable
# configuration. 
logp, cfg = most_probable_config(tn)

# ---

# Compute the most probable values of certain variables (e.g., 4 and 7) while
# marginalizing over others. This is known as Maximum a Posteriori (MAP)
# estimation.
mmap = MMAPModel(model, evidence=Dict(7=>0), queryvars=[4,7])

# ---

# Get the most probable configurations for variables 4 and 7.
most_probable_config(mmap)

# ---

# Compute the total log-probability of having lung cancer. The results suggest
# that the probability is roughly half.
log_probability(mmap, [1, 0]), log_probability(mmap, [0, 0])

# [^lauritzen1988local]:
#     Steffen L Lauritzen and David J Spiegelhalter. Local computations with probabilities on graphical structures and their application to expert systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 50(2):157–194, 1988.
