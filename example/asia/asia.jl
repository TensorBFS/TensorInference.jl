using TensorInference

# Load the model that detailed in the README and `asia.uai`.
instance = uai_problem_from_file(joinpath(@__DIR__, "asia.uai"))
tnet = TensorNetworkModel(instance)

# Get the probabilities (PR)
probability(tnet)

# Get the marginal probabilities (MAR)
marginals(tnet) .|> first

# The corresponding variables are
get_vars(tnet)

# Set the evidence variables "X-ray" (7) to be positive.
set_evidence!(instance, 7=>0)

# Since the evidence variable may change the contraction order, we re-compute the tensor network.
tnet = TensorNetworkModel(instance)

# Get the maximum log-probabilities (MAP)
maximum_logp(tnet)

# To sample from the probability model
sample(tnet, 10)

# Get not only the maximum log-probability, but also the most probable conifguration
# In the most probable configuration, the most probable one is the patient smoke (3) and has lung cancer (4)
logp, cfg = most_probable_config(tnet)

# Get the maximum log-probabilities (MMAP)
# To get the probability of lung cancer, we need to marginalize out other variables.
mmap = MMAPModel(instance; marginalized=[1,2,3,5,6,8])
# We get the most probable configurations on [4, 7]
most_probable_config(mmap)
# The total probability of having lung cancer is roughly half.
log_probability(mmap, [1, 0])
log_probability(mmap, [0, 0])