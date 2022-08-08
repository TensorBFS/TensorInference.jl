"""
$(TYPEDSIGNATURES)

Parse the problem instance found in `uai_filepath` defined in the UAI model
format.

The UAI file formats are defined in:
https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html
"""
function read_uai_file(uai_filepath; factor_eltype = Float64)

  # Read the uai file into an array of lines
  rawlines = open(uai_filepath) do file
    readlines(file)
  end

  # Filter out empty lines
  lines = filter(!isempty, rawlines)

  nvars   = lines[2] |> x -> parse.(Int, x)
  cards   = lines[3] |> split |> x -> parse.(Int, x)
  ntables = lines[4] |> x -> parse.(Int, x)

  scopes =
    lines[5:(5+ntables-1)] |>             # extract the factor scope definition lines
    x -> map(y -> split(y), x) |>         # split each line using blank space as delimeter
    x -> map(y -> map(z -> parse(Int, z), y), x) |> # parse each string element as an integer
    x -> map(y -> y[2:end], x) |>         # drop first element of each inner array
    x -> map(y -> map( z -> z +1, y), x) |> # convert to 1-based index
    x -> map(reverse, x)                  # order vars in ascending order (least significant first)

  parsed_margs =
    lines[(5+ntables):end] |>             # extract the probability tables definition lines
    x -> map(y -> y * " ", x) |>          # append a "space" to the end of each element
    x -> reduce(*, x) |>                  # concatenate all string elements
    x -> split(x)                         # split the array using blank space as delimeter

  tables2 = Array{factor_eltype,1}[]

  let i = 1
    while i <= length(parsed_margs)
      nelements = parsed_margs[i] |> x -> parse(Int, x)
      parsed_margs[i+1:i+nelements] |> x -> parse.(factor_eltype, x) |> x -> push!(tables2, x)
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
  factors = [Factor{factor_eltype,length(scope)}(Tuple(scope), table) for (scope, table) in zip(scopes_sorted, tables_sorted)]

  return nvars, cards, ntables, factors

end

"""
$(TYPEDSIGNATURES)

Return the observed variables and values in `uai_evid_filepath`. If the passed
file path is an empty string, return empty vectors.

The UAI file formats are defined in:
https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html
"""
function read_uai_evid_file(uai_evid_filepath::AbstractString)

  if isempty(uai_evid_filepath)
    # No evidence
    obsvars = Int64[]
    obsvals = Int64[]
  else
    # Read the uai evid file into an array of lines
    line = open(uai_evid_filepath) do file
      readlines(file)
    end

    @assert length(line) == 1

    # Extract number of observed vars, and their id together with their corresponding value
    nobsvars, rest = split(line[1]) |> x -> parse.(Int, x) |> x -> (x[1], x[2:end])
    observations = reshape(rest, 2, :)

    # Convert to 1-based indexing
    obsvars = observations[1,:] .+ 1
    obsvals = observations[2,:] .+ 1

    @assert nobsvars == length(obsvars)
  end

  # # DEBUG:
  # print("  "); @show obsvars
  # print("  "); @show obsvals

  return obsvars, obsvals

end

"""
$(TYPEDSIGNATURES)

Return the marginals of all variables. The order of the variables is the same
as in the model

The UAI file formats are defined in:
https://personal.utdallas.edu/~vibhav.gogate/uai16-evaluation/uaiformat.html
"""
function read_uai_mar_file(uai_mar_filepath::AbstractString; factor_eltype = Float64)

  # Read the uai mar file into an array of lines
  rawlines = open(uai_mar_filepath) do file
    readlines(file)
  end

  parsed_margs = 
    split(rawlines[2]) |>
    x -> x[2:end] |>
    x -> parse.(factor_eltype, x)

  marginals = Array{factor_eltype,1}[]

  let i = 1
    while i <= length(parsed_margs)
      nvars = parsed_margs[i] |> x -> convert(Int, x)
      parsed_margs[i+1:i+nvars] |> x -> push!(marginals, x)
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
  bags = lines[2:(2+nbags-1)] |>
    x -> map(split, x) |>
    x -> map(y -> y[3:end], x) |>
    x -> map(y -> parse.(Int, y), x)

  @assert length(bags) == nbags

  # # DEBUG:
  # @show bags

  # Parse edges and store then in a vector of vectors
  edges = lines[(2+nbags):end] |> 
    x -> map(split, x) |>
    x -> map(y -> parse.(Int, y), x)

  @assert length(edges) == nbags - 1

  # # DEBUG:
  # @show edges

  return nbags, treewidth, nvertices, bags, edges

end

"""
$(TYPEDSIGNATURES)

Read a UAI instance from a folder.
"""
function read_uai_dir(problem_dir::String, problem_filename::String)::UAIInstance
    uai_filepath = joinpath(problem_dir, problem_filename * ".uai")
    uai_evid_filepath = joinpath(problem_dir, problem_filename * ".uai.evid")
    uai_mar_filepath = joinpath(problem_dir, problem_filename * ".uai.MAR")

    reference_marginals = read_uai_mar_file(uai_mar_filepath)
    obsvars, obsvals = read_uai_evid_file(uai_evid_filepath)
    nvars, cards, nclique, factors = read_uai_file(uai_filepath; factor_eltype=Float64)
    return UAIInstance(nvars, nclique, cards, factors, obsvars, obsvals, reference_marginals)
end