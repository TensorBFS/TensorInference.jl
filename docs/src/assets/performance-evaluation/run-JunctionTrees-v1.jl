using ArgParse, JunctionTrees, BenchmarkTools, DataFrames, CSV, Artifacts

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
  td_filepath = joinpath(artifact"uai2014-mar", problem * ".tamaki.td")

  # -------------------------------- JunctionTrees --------------------------

  algo = compile_algo(
    uai_filepath,
    uai_evid_filepath=uai_evid_filepath,
    td_filepath=td_filepath,
  )

  eval(algo)
  obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)
  margs = Base.invokelatest(run_algo, obsvars, obsvals)
  jt_execution_time = @belapsed run_algo($obsvars, $obsvals)

  # ---------------------------------------------------------------------------

  df = DataFrame(
    problem = problem,
    library = "JunctionTrees_v1",
    execution_time = jt_execution_time,
  )
  CSV.write(output_file, df; append=true)

end

main()
