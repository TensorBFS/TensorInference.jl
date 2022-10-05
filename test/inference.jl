using Test, Artifacts
using OMEinsum
using TensorInference

@testset "gradient-based tensor network solvers" begin

  @testset "UAI 2014 problem set" begin

    benchmarks = [
                  "Alchemy", # fails
                  "CSP",
                  "DBN",
                  "Grids",
                  "linkage",
                  "ObjectDetection",
                  "Pedigree",
                  "Promedus",
                  "relational",
                  "Segmentation",
                 ]

    for benchmark in benchmarks

      @testset "$(benchmark) benchmark" begin

        # Capture the problem names that belong to the current benchmark
        rexp = Regex("($(benchmark)_\\d*)(\\.uai)\$") 
        problems = readdir(artifact"MAR_prob"; sort=false) |> 
          x -> map(y -> match(rexp, y), x) |> # apply regex
          x -> filter(!isnothing, x) |> # filter out `nothing` values
          x -> map(first, x) # get the first capture of each element

        for problem in problems

          @testset "$(problem)" begin

            problem = read_uai_problem(problem)

            # does not optimize over open vertices
            tn = TensorNetworkModeling(problem; optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40))
            # @info(tn)
            # @info timespace_complexity(tn)
            marginals2 = marginals(tn)
            # for dangling vertices, the output size is 1.
            npass = 0
            for i=1:problem.nvars
                npass += (length(marginals2[i]) == 1 && problem.reference_marginals[i] == [0.0, 1]) || isapprox(marginals2[i], problem.reference_marginals[i]; atol=1e-6)
            end
            @test npass == problem.nvars

          end
        end
      end
    end
  end

  # problem_name = "Promedus_14"
  # @testset "$problem_name" begin
  #   problem = read_uai_problem(problem_name)
  #   # does not optimize over open vertices
  #   tn = TensorNetworkModeling(problem; optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40))
  #   # @info(tn)
  #   # @info timespace_complexity(tn)
  #   marginals2 = marginals(tn)
  #   # for dangling vertices, the output size is 1.
  #   npass = 0
  #   for i=1:problem.nvars
  #       npass += (length(marginals2[i]) == 1 && problem.reference_marginals[i] == [0.0, 1]) || isapprox(marginals2[i], problem.reference_marginals[i]; atol=1e-6)
  #   end
  #   @test npass == problem.nvars
  # end

end 
