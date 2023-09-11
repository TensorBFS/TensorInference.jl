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

  # ------------------------------------ libdai --------------------------------

  libdai_binary = joinpath(homedir(), "repos", gethostname(), "libdai/examples/uai2010-aie-solver")
  temp_files_dir = joinpath(homedir(), "temp/marginals/libdai/"); mkpath(temp_files_dir)
  libdai_output_filepath = joinpath(temp_files_dir, problem*".uai")
  libdai_out = read(`$libdai_binary $uai_filepath $uai_evid_filepath $libdai_output_filepath`, String) |> x -> split(x, '\n')
  libdai_execution_time = filter(y -> startswith(y, "Total process time: "), libdai_out)[1] |> split |> x -> x[4] |> x -> parse(Float64, x) |> x -> x/1000

  # Log to file
  io = open(joinpath(dirname(output_file), "libdai.log.txt"), "w+")
  logger = SimpleLogger(io)
  with_logger(logger) do
    @info(libdai_out)
  end

  # ---------------------------------------------------------------------------

  df = DataFrame(
    problem = problem,
    library = "libdai",
    execution_time = libdai_execution_time,
  )
  CSV.write(output_file, df; append=true)

end

main()
