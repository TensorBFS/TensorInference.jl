module BenchTest

using BenchmarkTools

itr = (y for x in 1:1000 for y in 1:x)

const SUITE = BenchmarkGroup()

SUITE["test"] = @benchmarkable inv(rand(1000, 1000))

end  # module
BenchTest.SUITE
