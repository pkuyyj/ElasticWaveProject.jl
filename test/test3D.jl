using Test

print("Test 3D\n")

include("../src/wave3D_multixpu.jl")

@testset "Unit Test" begin
    # Test 1
    # physics
    lx, ly, lz  = 120.0, 120.0, 120.0
    stress      = 500
    source_radius = 2
    nx          = 127
    ny, nz      = nx, nx
    
    # derived numerics
    dx      = lx / nx
    dy      = ly / ny
    dz      = lz / nz
    xc, yc, zc = LinRange(-lx/2, lx/2, nx), LinRange(-ly/2, ly/2, ny), LinRange(-lz, 0, nz) # maybe z bottom is -lz, top is 0
    # init
    Sxx           = Data.Array([stress * exp((-xc[ix]^2 - yc[iy]^2 - (zc[iz] + lz / 2)^2)/ (2 * source_radius^2)) for ix = 1:nx, iy = 1:ny, iz = 1:nz])
    
    @test @inn ≈ Sxx_ori  # Use ≈ (isapprox) for comparing floating point numbers
end

"""
```julia
Sxx = elastic3D()
print(Sxx[10,20,30:40])
```
"""
## Sxx[10,20,30:40] with nx = 127, lx=120.0, nt=20000, nout=200
Sxx_truth = [-0.3259276094447621, -0.22555298039829758, -0.1068580840608831, 0.04278624627954669, 0.221729179931364, 0.41323512500050646, 0.5892924805028406, 0.7262773515637876, 0.8174373435556722, 0.8701234248499624, 0.8905463133044659][-1.3696442351595288, -1.4762115101596556, -1.475465441989736, -1.3699505158048375, -1.187879336049613, -0.9813634679799219, -0.8042569558408565, -0.6919069570748017, -0.6528806870797416, -0.6808620187616686, -0.7701429778885155][3.5839591021953745, 3.712930049246794, 3.797363874517482, 3.885365892321424, 4.0277734306469455, 4.209002289899644, 4.3518829217449255, 4.413246555846969, 4.427045121318717, 4.44983697096112, 4.485914017177473][-0.08039282906694742, -0.15654081278850276, -0.24323685728356315, -0.3299953958934951, -0.4125445743451796, -0.4881281840732669, -0.5627726022670507, -0.6396093245494433, -0.7066392544487696, -0.7356813034891518, -0.7075175999338034]

@testset "Reference Tests for diffusion_3D" begin
    Sxx = elastic3D()

    ## Test that selected value of T is the same as Sxx_truth
    @test Sxx[10,20,30:40] ≈ Sxx_truth

end
