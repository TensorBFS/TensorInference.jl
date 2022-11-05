module BenchTest

using BenchmarkTools

const SUITE = BenchmarkGroup()

SUITE["test"] = @benchmarkable inv(rand(1000, 1000))

end  # module
BenchTest.SUITE
