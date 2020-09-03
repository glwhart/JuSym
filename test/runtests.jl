using MinkReduction
using Test

@testset "MinkReduction.jl" begin
    @test m = DeviousMat(10); a = m[1,:]; b = m[2,:]; c = m[3,:]; d,e,f = minkReduce(a,b,c); hcat(d,e,f) == [0.0 -1.0 0.0; 0.0 0.0 -1.0; 1.0 0.0 0.0]

end
