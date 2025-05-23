"""
$(TYPEDSIGNATURES)

Parse the problem instance found in `model_filepath` defined in the UAI model
format. If the provided file path is empty, return `nothing`.

The UAI file formats are defined in:
https://uaicompetition.github.io/uci-2022/file-formats/
"""
function read_model_file(model_filepath::AbstractString; factor_eltype = Float64)::UAIModel
    # Read the uai file into an array of lines
    str = open(model_filepath) do file
        read(file, String)
    end
    return read_model_from_string(str; factor_eltype)
end

function read_model_from_string(str::AbstractString; factor_eltype = Float64)::UAIModel
    rawlines = split(str, "\n")
    # Filter out empty lines
    lines = filter(!isempty, rawlines)

    nvars   = lines[2] |> x -> parse.(Int, x)
    cards   = lines[3] |> split |> x -> parse.(Int, x)
    ntables = lines[4] |> x -> parse.(Int, x)

    scopes =
        lines[5:(5 + ntables - 1)] |> # extract the factor scope definition lines
        x -> map(y -> split(y), x) |> # split each line using blank space as delimeter
             x -> map(y -> map(z -> parse(Int, z), y), x) |> # parse each string element as an integer
                  x -> map(y -> y[2:end], x) |> # drop first element of each inner array
                       x -> map(y -> map(z -> z + 1, y), x) |> # convert to 1-based index
                            x -> map(reverse, x) # order vars in ascending order (least significant first)

    parsed_margs =
        lines[(5 + ntables):end] |> # extract the probability tables definition lines
        x -> map(y -> y * " ", x) |> # append a "space" to the end of each element
             x -> reduce(*, x) |> # concatenate all string elements
                  x -> split(x) # split the array using blank space as delimeter

    tables2 = Array{factor_eltype, 1}[]

    let i = 1
        while i <= length(parsed_margs)
            nelements = parsed_margs[i] |> x -> parse(Int, x)
            parsed_margs[(i + 1):(i + nelements)] |> x -> parse.(factor_eltype, x) |> x -> push!(tables2, x)
            i += nelements + 1
        end
    end

    tables =
        zip(tables2, map(scope -> cards[scope], scopes)) |> # pair each table with its card vector
        x -> map(y -> reshape(y[1], Tuple(y[2])), x) # reshape each factor according to its card

    # Sort scope vars in ascending order and permute table dims accordingly
    scopes_sorted = map(sort, scopes)
    tables_sorted = map(indexin, scopes_sorted, scopes) |> x -> map(permutedims, tables, x)

    # Wrap the tables with their corresponding scopes in an array of Factor type
    factors = [Factor{factor_eltype, length(scope)}(Tuple(scope), table) for (scope, table) in zip(scopes_sorted, tables_sorted)]

    return UAIModel(nvars, cards, factors)
end

"""
$(TYPEDSIGNATURES)

Return the observed variables and values in `evidence_filepath`. If the passed
file path is an empty string, return empty vectors.

The UAI file formats are defined in:
https://uaicompetition.github.io/uci-2022/file-formats/
"""
function read_evidence_file(evidence_filepath::AbstractString)

    isempty(evidence_filepath) && return Int64[], Int64[] # no evidence

    # Read the last line of the uai evid file
    line = open(evidence_filepath) do file
        readlines(file)
    end |> last

    # Extract number of observed vars, and their id together with their corresponding value
    nobsvars, rest = split(line) |> x -> parse.(Int, x) |> x -> (x[1], x[2:end])
    observations = reshape(rest, 2, :)

    # Convert to 1-based indexing
    obsvars = observations[1, :] .+ 1
    obsvals = observations[2, :]

    @assert nobsvars == length(obsvars)

    return obsvars, obsvals
end

"""
$(TYPEDSIGNATURES)

Return the query variables in `query_filepath`. If the passed file path is an
empty string, return an empty vector.

The UAI file formats are defined in:
https://uaicompetition.github.io/uci-2022/file-formats/
"""
function read_query_file(query_filepath::AbstractString)
    isempty(query_filepath) && return Int64[]

    # Read the first line of the uai query file
    line = open(query_filepath) do file
        readlines(file)
    end |> first

    # Separate the number of query vars and their indices
    nqueryvars, queryvars_zero_based = split(line) |> x -> parse.(Int, x) |> x -> (x[1], x[2:end])

    # Convert to 1-based indexing
    queryvars = queryvars_zero_based .+ 1

    @assert nqueryvars == length(queryvars)

    return queryvars
end

"""
$(TYPEDSIGNATURES)

Parse the solution marginals of all variables from the UAI MAR solution file.
The order of the variables is the same as in the model definition.

The UAI file formats are defined in:
https://uaicompetition.github.io/uci-2022/file-formats/
"""
function parse_mar_solution_file(rawlines::Vector{String}; factor_eltype = Float64)

    parsed_margs = split(rawlines[2]) |> x -> x[2:end] |> x -> parse.(factor_eltype, x)

    marginals = Array{factor_eltype, 1}[]

    let i = 1
        while i <= length(parsed_margs)
            nvars = parsed_margs[i] |> x -> convert(Int, x)
            parsed_margs[(i + 1):(i + nvars)] |> x -> push!(marginals, x)
            i += nvars + 1
        end
    end

    return marginals
end

"""
$(TYPEDSIGNATURES)

Parse a tree decomposition instance described the PACE format.

The PACE file format is defined in:
https://pacechallenge.org/2017/treewidth/
"""
function read_td_file(td_filepath::AbstractString)

    # Read the td file into an array of lines
    rawlines = open(td_filepath) do file
        readlines(file)
    end

    # Filter out comments
    lines = filter(x -> !startswith(x, "c"), rawlines)

    # Extract number of bags, treewidth+1 and number of vertices from solution line
    nbags, treewidth, nvertices = split(lines[1]) |> x -> x[3:5] |> x -> parse.(Int, x)

    # Parse bags and store then in a vector of vectors
    bags = lines[2:(2 + nbags - 1)] |>
           x -> map(split, x) |>
                x -> map(y -> y[3:end], x) |>
                     x -> map(y -> parse.(Int, y), x)

    @assert length(bags) == nbags

    # Parse edges and store then in a vector of vectors
    edges = lines[(2 + nbags):end] |> x -> map(split, x) |> x -> map(y -> parse.(Int, y), x)

    @assert length(edges) == nbags - 1

    return nbags, treewidth, nvertices, bags, edges
end

# patch to get content by broadcasting into array, while keep array size unchanged.
broadcasted_content(x) = asarray(content.(x), x)

"""
$TYPEDEF

Specify the UAI models from the artifacts.
It can be used as the input of [`read_model`](@ref).

### Fields
$TYPEDFIELDS
"""
struct ArtifactProblemSpec
    artifact_path::String
    task::String
    problem_set::String
    problem_id::Int
end

"""
$TYPEDSIGNATURES

Get artifact from artifact name, task name, problem set name and problem id.
"""
function problem_from_artifact(artifact_name::String, task::String, problem_set::String, problem_id::Int)
    path = get_artifact_path(artifact_name)
    return ArtifactProblemSpec(path, task, problem_set, problem_id)
end


"""
$TYPEDSIGNATURES

Read an UAI model from an artifact.
"""
function read_model(problem::ArtifactProblemSpec; eltype=Float64)
    problem_name = "$(problem.problem_set)_$(problem.problem_id).uai"
    return read_model_file(joinpath(problem.artifact_path, problem.task, problem_name); factor_eltype = eltype)
end

"""
$(TYPEDSIGNATURES)

Return the solution in the artifact.

The UAI file formats are defined in:
https://uaicompetition.github.io/uci-2022/file-formats/
"""
function read_solution(problem::ArtifactProblemSpec; factor_eltype=Float64)
    problem_name = "$(problem.problem_set)_$(problem.problem_id).uai.$(problem.task)"
    solution_filepath = joinpath(problem.artifact_path, problem.task, problem_name)

    # Read the solution file into an array of lines
    rawlines = open(solution_filepath) do file
        readlines(file)
    end

    if problem.task == "MAR" || problem.task == "MAR2"
        return parse_mar_solution_file(rawlines; factor_eltype)
    elseif problem.task == "MAP" || problem.task == "MMAP"
        # Return all elements except the first in the last line as a vector of integers
        return last(rawlines) |> split |> x -> x[2:end] |> x -> parse.(Int, x)
    elseif problem.task == "PR"
        # Parse the number in the last line as a floating point
        return last(rawlines) |> x -> parse(Float64, x)
    end
end

"""
$TYPEDSIGNATURES
"""
function read_evidence(problem::ArtifactProblemSpec)
    problem_name = "$(problem.problem_set)_$(problem.problem_id).uai.evid"
    evidence_filepath = joinpath(problem.artifact_path, problem.task, problem_name)
    obsvars, obsvals = read_evidence_file(evidence_filepath)
    return Dict(zip(obsvars, obsvals))
end

"""
$TYPEDSIGNATURES
"""
function read_queryvars(problem::ArtifactProblemSpec)
    problem_name = "$(problem.problem_set)_$(problem.problem_id).uai.query"
    query_filepath = joinpath(problem.artifact_path, problem.task, problem_name)
    return read_query_file(query_filepath)
end

"""
$TYPEDSIGNATURES

Helper function that captures the problem names that belong to `problem_set`
for the given task.
"""
function dataset_from_artifact(artifact_name::AbstractString)
    artifact_path = get_artifact_path(artifact_name)
    tasks = ["PR", "MAR", "MAR2", "MAP", "MMAP"]
    problems = Dict{String, Dict{String, Dict{Int, ArtifactProblemSpec}}}()

    regex = r"^([a-zA-Z_{1}][a-zA-Z0-9_]+)_(\d+)\.uai$"
    for task in tasks
        problems_task = Dict{String, Dict{Int, ArtifactProblemSpec}}()
        problems[task] = problems_task
        readdir(joinpath(artifact_path, task); sort = false) |>
           x -> map(y -> match(regex, y), x) |> # apply regex
                x -> filter(!isnothing, x) |> # filter out `nothing` values
                     x -> map(x) do m   # matched the `problem_set` and `problem_id`
                        problem_set, problem_id = m[1], parse(Int, m[2])
                        haskey(problems_task, problem_set) || (problems_task[problem_set] = Dict{Int, ArtifactProblemSpec}())
                        set = problems_task[problem_set]
                        haskey(set, problem_id) || (set[problem_id] = ArtifactProblemSpec(artifact_path, task, problem_set, problem_id))
                     end
    end
    return problems
end

function get_artifact_path(artifact_name::String)
    artifact_toml = pkgdir(TensorInference, "Artifacts.toml")
    Pkg.ensure_artifact_installed(artifact_name, artifact_toml)
    artifact_hash = Pkg.Artifacts.artifact_hash(artifact_name, artifact_toml)
    return Pkg.Artifacts.artifact_path(artifact_hash)
end

togpu(x) = error("You must import CUDA with `using CUDA` before using GPU!")

"""
$TYPEDSIGNATURES

Matrix product state (MPS) is a tensor network model that is widely used in
quantum many-body physics. It is a special case of tensor network model where
the tensors are rank-3 tensors and the physical indices are connected in a
chain. The MPS is defined as:

```math
\\begin{align*}
\\left| \\psi \\right\\rangle &= \\sum_{x_1, x_2, \\ldots, x_n} \\text{Tr}(A_1^{x_1} A_2^{x_2} \\cdots A_n^{x_n}) \\left| x_1, x_2, \\ldots, x_n \\right\\rangle \\\\
\\left\\langle \\psi \\right| &= \\sum_{x_1, x_2, \\ldots, x_n} \\text{Tr}(A_n^{x_n} \\cdots A_2^{x_2} A_1^{x_1}) \\left\\langle x_1, x_2, \\ldots, x_n \\right|
\\end{align*}
```

where \$A_i^{x_i}\$ is a rank-3 tensor with physical index \$x_i\$ and two virtual
indices connecting to the next tensor. The MPS is a special case of the tensor
network model where the tensors are rank-3 tensors and the physical indices are
connected in a chain.

### Arguments
- `n` is the number of physical indices.
- `chi` is the bond dimension of the virtual indices.
- `d` is the dimension of the physical indices.
"""
function random_matrix_product_state(::Type{T}, n::Int, chi::Int, d::Int=2) where T
    uai = random_matrix_product_uai(T, n, chi, d)
    return TensorNetworkModel(uai; optimizer=GreedyMethod())
end
random_matrix_product_state(n::Int, chi::Int, d::Int=2) = random_matrix_product_state(ComplexF64, n, chi, d)

"""
$TYPEDSIGNATURES

Generate a random UAIModel that represents a matrix product state (MPS).
Similar to [`random_matrix_product_state`](@ref), but returns the UAIModel directly.
"""
function random_matrix_product_uai(::Type{T}, n::Int, chi::Int, d::Int=2) where T
    # chi ^ (n-1) * (variance^n)^2 == 1/d^n
    variance = d^(-1/2) * chi^(-1/2+1/2n)
    tensors = Any[randn(T, d, chi) .* variance]
    physical_indices = collect(1:n)
    virtual_indices_ket = collect(n+1:2n-1)
    virtual_indices_bra = collect(2n:3n-2)
    ixs_ket = [[physical_indices[1], virtual_indices_ket[1]]]
    ixs_bra = [[physical_indices[1], virtual_indices_bra[1]]]
    for i = 2:n-1
        push!(tensors, randn(T, chi, d, chi) .* variance)
        push!(ixs_ket, [virtual_indices_ket[i-1], physical_indices[i], virtual_indices_ket[i]])
        push!(ixs_bra, [virtual_indices_bra[i-1], physical_indices[i], virtual_indices_bra[i]])
    end
    push!(tensors, randn(T, chi, d) .* variance)
    push!(ixs_ket, [virtual_indices_ket[n-1], physical_indices[n]])
    push!(ixs_bra, [virtual_indices_bra[n-1], physical_indices[n]])
    tensors, ixs = [tensors..., conj.(tensors)...], [ixs_ket..., ixs_bra...]
    size_dict = OMEinsum.get_size_dict(ixs, tensors)
    nvars = 3n-2
    return UAIModel(
        nvars,
        [size_dict[i] for i=1:nvars],
        [Factor((ixs[i]...,), tensors[i]) for i in 1:length(tensors)]
    )
end


"""
$TYPEDSIGNATURES

Tensor train (TT) is a tensor network model that is widely used in quantum
many-body physics. This model is different from the matrix product state (MPS)
in that it does not have an extra copy for representing the bra state.
"""
function random_tensor_train_uai(::Type{T}, n::Int, chi::Int, d::Int=2; periodic=false) where T
    # chi ^ (n-1) * (variance^n)^2 == 1/d^n
    variance = d^(-1/2) * chi^(-1/2+1/2n)
    physical_indices = collect(1:n)
    virtual_indices = collect(n+1:2n)
    tensors = Any[(periodic ? rand(T, chi, d, chi) : rand(T, d, chi)) .* variance]
    ixs = [periodic ? [virtual_indices[n], physical_indices[1], virtual_indices[1]] : [physical_indices[1], virtual_indices[1]]]
    for i = 2:n-1
        push!(tensors, rand(T, chi, d, chi) .* variance)
        push!(ixs, [virtual_indices[i-1], physical_indices[i], virtual_indices[i]])
    end
    push!(tensors, (periodic ? rand(T, chi, d, chi) : rand(T, chi, d)) .* variance)
    push!(ixs, periodic ? [virtual_indices[n-1], physical_indices[n], virtual_indices[n]] : [virtual_indices[n-1], physical_indices[n]])
    size_dict = OMEinsum.get_size_dict(ixs, tensors)
    nvars = periodic ? 2n : 2n-1
    return UAIModel(
        nvars,
        [size_dict[i] for i=1:nvars],
        [Factor((ixs[i]...,), tensors[i]) for i in 1:length(tensors)]
    )
end