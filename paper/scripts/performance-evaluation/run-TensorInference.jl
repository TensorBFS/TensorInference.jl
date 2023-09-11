using ArgParse, TensorInference, BenchmarkTools, DataFrames, CSV, Artifacts

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

  problem_name = parsed_args["problem"]
  output_file = parsed_args["output-file"]

  # -------------------------------- TensorInference --------------------------

  optimizers = Dict(
    "Alchemy" => TreeSA(ntrials=5, niters=5, βs=0.1:0.1:100),
    "CSP" => TreeSA(ntrials=5, niters=5, βs=0.1:0.1:100),
    "DBN" => KaHyParBipartite(sc_target=25),
    "Grids" => TreeSA(ntrials=5, niters=5, βs=0.1:0.1:100),
    "linkage" => TreeSA(ntrials=5, niters=20, βs=0.1:0.1:40),
    "ObjectDetection" => TreeSA(ntrials=5, niters=5, βs=1:0.1:100),
    "Pedigree" => TreeSA(ntrials=5, niters=5, βs=0.1:0.1:100),
    "Promedus" => TreeSA(ntrials=5, niters=5, βs=0.1:0.1:100),
    "relational" => TreeSA(ntrials=5, niters=5, βs=0.1:0.1:100),
    "Segmentation" => TreeSA(ntrials=5, niters=5, βs=0.1:0.1:100)
  )

  problem_set, problem_number = split(problem_name, "_") |> x -> (first(x), parse(Int, last(x)))
  problem = problem_from_artifact("uai2014", "MAR", problem_set, problem_number)

  tn = TensorNetworkModel(
    read_model(problem);
    optimizer = optimizers[problem_set],
    evidence=read_evidence(problem)
  )
  ti_execution_time = @belapsed marginals($tn)

  # ---------------------------------------------------------------------------

  df = DataFrame(
    problem = problem_name,
    library = "TensorInference",
    execution_time = ti_execution_time,
  )
  CSV.write(output_file, df; append=true)

end

main()
