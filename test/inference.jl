using Test, Artifacts
using OMEinsum
using KaHyPar
using TensorInference

@testset "gradient-based tensor network solvers" begin

  @testset "UAI 2014 problem set" begin

    # TODO: rescale while contract.
    benchmarks = [
                  ("Alchemy", [1/3], TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),
                  ("CSP",fill(1.0, 3), TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),
                  ("DBN",fill(1.0, 6), KaHyParBipartite(sc_target=25)),
                  ("Grids", [1/4, 1/10, 1/10, 1/32, 1/2, 1/4, 1/15, 1/44], TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)), # greedy also works
                  ("linkage", fill(1.2, 17), TreeSA(ntrials=3, niters=20, βs=0.1:0.1:40)),  # linkage_15 fails
                  ("ObjectDetection",fill(1.5, 65), TreeSA(ntrials=1, niters=5, βs=1:0.1:100)),  # ObjectDetection_35 fails
                  ("Pedigree",fill(1.0, 3), TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)), # greedy also works
                  ("Promedus",fill(1.0, 28), TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),  # greedy also works
                  #("relational",fill(1.0, 5), TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),
                  ("Segmentation", fill(1.0, 6), TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),  # greedy also works
                 ]
    #benchmarks = [("relational", fill(1.0, 5), TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100))]
    #benchmarks = [("DBN",fill(1.0, 6), SABipartite(sc_target=25, βs=0.1:0.01:50))]
    for (benchmark, rescales, optimizer) in benchmarks

      @testset "$(benchmark) benchmark" begin

        # Capture the problem names that belong to the current benchmark
        rexp = Regex("($(benchmark)_\\d*)(\\.uai)\$") 
        problems = readdir(artifact"MAR_prob"; sort=false) |> 
          x -> map(y -> match(rexp, y), x) |> # apply regex
          x -> filter(!isnothing, x) |> # filter out `nothing` values
          x -> map(first, x) # get the first capture of each element
        @assert length(problems) == length(rescales)

        for (problem, rescale) in [zip(problems, rescales)...]
          @show problem

          @testset "$(problem)" begin

            problem = read_uai_problem(problem)

            # does not optimize over open vertices
            #tn = TensorNetworkModeling(problem; optimizer=, rescale)
            tn = TensorNetworkModeling(problem; optimizer, rescale)
            sc = contraction_complexity(tn).sc
            if sc > 28
              error("space complexity too large! got $(sc)")
            end
            # @info(tn)
            # @info timespace_complexity(tn)
            marginals2 = marginals(tn)
            # for dangling vertices, the output size is 1.
            npass = 0
            for i=1:problem.nvars
                npass += length(marginals2[i]) == 1 || isapprox(marginals2[i], problem.reference_marginals[i]; atol=1e-6)
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
