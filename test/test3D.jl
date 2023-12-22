using Test

include("../src/wave3D_multixpu.jl")

"""
```julia
Sxx = elastic3D(nx = 31, lx=120.0, nt=200)
print(Sxx[10,20,10:20])
```
"""
## Sxx[10,20,10:20] with nx = 31, lx=120.0, nt=200, nout=200
Sxx_truth = [-0.015046320519832717, 0.5470902195941995, 0.5294086126094854, 0.2932688372267475, 0.12386706824395867, 0.03708124842316666, 0.008947533009905449, 0.0017134803455876972, 0.0002775473019659045, 3.83172672383034e-5, 4.650961000122782e-6]

@testset "Reference Tests for elastic_3D" begin
    Sxx = elastic3D(nx = 31, lx=120.0, nt=200)
    ## Test that selected value of Sxx is the same as Sxx_truth
    @test Sxx[10,20,10:20] ≈ Sxx_truth

end
