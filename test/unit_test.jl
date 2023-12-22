using Test


include("../src/wave3D_multixpu.jl")

@testset "Unit Test compute_V" begin
    # Test 1
    # physics
    lx          = 120.0
    nx          = 31

    ly, lz     = lx, lx
    λ, μ       = 4.0, 2.0           # Lame parameters (λ, μ)
    k          = 1.0                # bulk modulus
    ρ          = 2.0                # density
    t          = 0.0                # physical time
    damping_sxx  = 1.0              # 1 - damping factor
    damping_sxy  = 0.95             # 1 - damping factor
    ## Numerics
    ny, nz     = nx, nx             # numerical grid resolution
    ## Derived numerics
    me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz) # MPI initialisation
    dx, dy, dz = lx/(nx_g()-1), ly/(ny_g()-1), lz/(nz_g()-1)      # cell sizes
    ## Array allocations
    Vx         = @zeros(nx+1,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    Vz         = @zeros(nx  ,ny  ,nz+1)
    Sxx        = @zeros(nx  ,ny  ,nz  )
    Syy        = @zeros(nx  ,ny  ,nz  )
    Szz        = @zeros(nx  ,ny  ,nz  )
    Sxy        = @zeros(nx  ,ny  ,nz  )
    Sxz        = @zeros(nx  ,ny  ,nz  )
    Syz        = @zeros(nx  ,ny  ,nz  )
    ## Initial conditions
    dt         = 0.05 * min(dx,dy,dz)/sqrt(k/ρ)/6.1 # prevent grad explosion

    ## Initialize diagonal stress tensor components, can be customized
    stress      = 500
    source_radius = 2
    Sxx .= Data.Array([stress * exp((-(x_g(ix,dx,Sxx)-0.2*lx)^2 - (y_g(iy,dy,Sxx)-0.5*ly)^2 - (z_g(iz,dz,Sxx)-0.3*lz)^2)/ (2 * source_radius^2)) for ix=1:size(Sxx,1), iy=1:size(Sxx,2), iz=1:size(Sxx,3)])
    Szz .= Data.Array([stress * exp((-(x_g(ix,dx,Szz)-0.2*lx)^2 - (y_g(iy,dy,Szz)-0.5*ly)^2 - (z_g(iz,dz,Szz)-0.3*lz)^2)/ (2 * source_radius^2)) for ix=1:size(Szz,1), iy=1:size(Szz,2), iz=1:size(Szz,3)])
    @hide_communication (8, 8, 4) begin # communication/computation overlap
        @parallel compute_V!(Vx, Vy, Vz, Sxx,Syy,Szz,Sxy,Sxz,Syz, dt, ρ, dx, dy, dz)
        update_halo!(Vx, Vy, Vz)
    end
    Vx_ref = [-1.2311080817990726e-17, -1.6661236094516045e-18, -4.129907523523374e-21, -1.8749751149351893e-25, -1.5590956556731229e-31, -2.3744995255971116e-39, -6.623590663185542e-49, -3.384051438366393e-60, -3.1666677467393994e-73, -5.427378551028805e-88, -1.703726198439264e-104]
    @test Array(Vx)[10,20,10:20] ≈ Vx_ref # Use ≈ (isapprox) for comparing floating point numbers
    finalize_global_grid()
end