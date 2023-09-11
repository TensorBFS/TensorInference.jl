using ArgParse, JunctionTrees, BenchmarkTools, DataFrames, CSV, Artifacts, Logging

function parse_commandline()

  s = ArgParseSettings()

  @add_arg_table s begin
    "--problem"
    help = "a problem instance in the UAI format"
    "--output-file"
    help = "relative file path of the output file"
  end

  return parse_args(s)

end

function main()

  parsed_args = parse_commandline()

  problem = parsed_args["problem"]
  output_file = parsed_args["output-file"]

  uai_filepath = joinpath(artifact"uai2014-mar", problem * ".uai")
  uai_evid_filepath = joinpath(artifact"uai2014-mar", problem * ".uai.evid")

  # ------------------------------------ merlin --------------------------------

  merlin_binary = joinpath(homedir(), "repos", gethostname(), "merlin/bin/merlin")
  temp_files_dir = joinpath(homedir(), "temp/marginals/merlin/"); mkpath(temp_files_dir)
  merlin_output_filepath = joinpath(temp_files_dir, problem*".uai")
  merlin_out = read(`$merlin_binary --input-file $uai_filepath --evidence-file $uai_evid_filepath --task MAR --algorithm cte --output-file $merlin_output_filepath`, String) |> x -> split(x, '\n')
  merlin_total_time = filter(y -> startswith(y, "[CTE] Finished in "), merlin_out)[1] |> split |> x -> x[4] |> x -> parse(Float64, x)
  merlin_initialization_time = filter(y -> startswith(y, "[CTE] Finished initialization in "), merlin_out)[1] |> split |> x -> x[5] |> x -> parse(Float64, x)
  merlin_execution_time = merlin_total_time - merlin_initialization_time

  # Log to file
  io = open(joinpath(dirname(output_file), "merlin.log.txt"), "w+")
  logger = SimpleLogger(io)
  with_logger(logger) do
    @info(merlin_out)
  end

  # ---------------------------------------------------------------------------

  df = DataFrame(
    problem = problem,
    library = "merlin",
    execution_time = merlin_execution_time,
  )
  CSV.write(output_file, df; append=true)

end

main()
