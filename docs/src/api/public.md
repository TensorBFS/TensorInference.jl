# Public API

## Index

[Modules](@ref)

```@index
Pages = ["public.md"]
Order   = [:module]
```

[Types](@ref)

```@index
Pages = ["public.md"]
Order   = [:type]
```

[Functions](@ref)

```@index
Pages = ["public.md"]
Order   = [:function]
```

## Modules

```@docs
TensorInference
```

## Types

```@docs
GreedyMethod
KaHyParBipartite
MergeGreedy
MergeVectors
SABipartite
TreeSA
MMAPModel
RescaledArray
TensorNetworkModel
ArtifactProblemSpec
UAIModel
BeliefPropgation
```

## Functions

```@docs
contraction_complexity
get_cards
get_vars
log_probability
marginals
maximum_logp
most_probable_config
probability
belief_propagate
dataset_from_artifact
problem_from_artifact
read_model
read_evidence
read_solution
read_queryvars
read_model_file
read_evidence_file
read_td_file
sample
update_evidence!
update_temperature
random_matrix_product_state
random_matrix_product_uai
random_tensor_train_uai
```
