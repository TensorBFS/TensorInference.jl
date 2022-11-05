module BenchTest

using BenchmarkTools

const SUITE = BenchmarkGroup()

SUITE["test"] = @benchmarkable inv(rand(100, 100))

end  # module
BenchTest.SUITE
