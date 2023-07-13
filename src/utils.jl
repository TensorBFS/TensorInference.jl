"""
$(TYPEDSIGNATURES)

Parse the problem instance found in `model_filepath` defined in the UAI model
format.

The UAI file formats are defined in:
https://uaicompetition.github.io/uci-2022/file-formats/
"""
function read_model_file(model_filepath; factor_eltype = Float64)
    # Read the uai file into an array of lines
    str = open(model_filepath) do file
        read(file, String)
    end
    return read_model_string(str; factor_eltype)
end

function read_model_string(str; factor_eltype = Float64)
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

    return nvars, cards, ntables, factors
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

Return the solution in `solution_filepath`. Returns an empty vector if the
extension is not supported.

The UAI file formats are defined in:
https://uaicompetition.github.io/uci-2022/file-formats/
"""
function read_solution_file(solution_filepath::AbstractString; factor_eltype = Float64)

    result = Vector{factor_eltype}[]
    extension = splitext(solution_filepath)[2]

    # Read the solution file into an array of lines
    rawlines = open(solution_filepath) do file
        readlines(file)
    end

    if extension == ".MAR"
        result = parse_mar_solution_file(rawlines; factor_eltype)
    elseif extension == ".MAP" || extension == ".MMAP"
        # Return all elements except the first in the last line as a vector of integers
        result = last(rawlines) |> split |> x -> x[2:end] |> x -> parse.(Int, x)
    elseif extension == ".PR"
        # Parse the number in the last line as a floating point
        result = last(rawlines) |> x -> parse(Float64, x)
    end

    return result
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

    # # DEBUG:
    # @show nbags, treewidth, nvertices

    # Parse bags and store then in a vector of vectors
    bags = lines[2:(2 + nbags - 1)] |>
           x -> map(split, x) |>
                x -> map(y -> y[3:end], x) |>
                     x -> map(y -> parse.(Int, y), x)

    @assert length(bags) == nbags

    # # DEBUG:
    # @show bags

    # Parse edges and store then in a vector of vectors
    edges = lines[(2 + nbags):end] |> x -> map(split, x) |> x -> map(y -> parse.(Int, y), x)

    @assert length(edges) == nbags - 1

    # # DEBUG:
    # @show edges

    return nbags, treewidth, nvertices, bags, edges
end

"""
$(TYPEDSIGNATURES)

Read a UAI problem instance from a file.
"""
function read_instance(
    model_filepath::AbstractString;
    evidence_filepath::AbstractString = "",
    query_filepath::AbstractString = "",
    solution_filepath::AbstractString = "",
    eltype = Float64
)::UAIInstance
    nvars, cards, ncliques, factors = read_model_file(model_filepath; factor_eltype = eltype)
    obsvars, obsvals = read_evidence_file(evidence_filepath)
    queryvars = read_query_file(query_filepath)
    reference_solution = isempty(solution_filepath) ? Vector{eltype}[] : read_solution_file(solution_filepath)
    return UAIInstance(nvars, ncliques, cards, factors, obsvars, obsvals, queryvars, reference_solution)
end

function read_instance_from_string(uai::AbstractString; eltype = Float64)::UAIInstance
    nvars, cards, ncliques, factors = read_model_string(uai; factor_eltype = eltype)
    return UAIInstance(nvars, ncliques, cards, factors, Int[], Int[], Vector{eltype}[])
end

# patch to get content by broadcasting into array, while keep array size unchanged.
broadcasted_content(x) = asarray(content.(x), x)
