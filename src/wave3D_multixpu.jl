# import Pkg; Pkg.precompile()
# using Pkg
# Pkg.instantiate()
const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=false)
end
using ImplicitGlobalGrid, Plots, Printf, Statistics
default(size=(1400, 800), framestyle=:box, label=false, grid=false, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

import MPI

# CPU functions
@views av_zi(A) = (A[2:end-1,2:end-1,2:end-2] .+ A[2:end-1,2:end-1,3:end-1]).*0.5
@views av_za(A) = (A[:,:,1:end-1] .+ A[:,:,2:end]).*0.5
@views in_x(A)  = A[:end-1, :, :]
@views in_y(A)  = A[: , :end-1, :]
@views in_z(A)  = A[: , :, :end-1]
@views in_xy(A)  = A[:end-1, :end-1, :]
@views in_yz(A)  = A[:, :end-1, :end-1]
@views in_xz(A)  = A[:end-1, :, :end-1]

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, Sxx::Data.Array, Syy::Data.Array, Szz::Data.Array, Sxy::Data.Array, Sxz::Data.Array, Syz::Data.Array, dt::Data.Number, ρ::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @inn(Vx) = @inn(Vx) + dt/ρ * (@d_xi(Sxx)/dx + @d_yi(Sxy)/dy + @d_zi(Sxz)/dz)
    @inn(Vy) = @inn(Vy) + dt/ρ * (@d_xi(Sxy)/dx + @d_yi(Syy)/dy + @d_zi(Syz)/dz)
    @inn(Vz) = @inn(Vz) + dt/ρ * (@d_xi(Sxz)/dx + @d_yi(Syz)/dy + @d_zi(Szz)/dz)
    return
end

@parallel function compute_S!(Sxx::Data.Array, Syy::Data.Array, Szz::Data.Array, Sxy::Data.Array, Sxz::Data.Array, Syz::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dt::Data.Number, λ::Data.Number, μ::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number, damping_sxx::Data.Number)
    @all(Sxx) = damping_sxx * @all(Sxx) + dt * (λ * (@d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz) + 2μ * @d_xa(Vx)/dx)
    @all(Syy) = damping_sxx * @all(Syy) + dt * (λ * (@d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz) + 2μ * @d_ya(Vy)/dy)
    @all(Szz) = damping_sxx * @all(Szz) + dt * (λ * (@d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz) + 2μ * @d_za(Vz)/dz)
    return
end


@parallel_indices (ix, iy, iz) function compute_S_diag!(Sxx::Data.Array, Syy::Data.Array, Szz::Data.Array, Sxy::Data.Array, Sxz::Data.Array, Syz::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dt::Data.Number, λ::Data.Number, μ::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number, damping_sxy::Data.Number)
    if (ix <= size(Sxy, 1)-2 && iy <= size(Sxy, 2)-2 && iz <= size(Sxy, 3))
        Sxy[ix+1, iy+1, iz] = damping_sxy * Sxy[ix+1, iy+1, iz] + dt * μ * ((Vy[ix+2, iy+1, iz] - Vy[ix, iy+1, iz])/(2*dx) + (Vx[ix+1, iy+2, iz] - Vx[ix+1, iy, iz])/(2*dy))
    end
    if (ix <= size(Sxz, 1)-2 && iy <= size(Sxz, 2) && iz <= size(Sxz, 3)-2)
        Sxz[ix+1, iy, iz+1] = damping_sxy * Sxz[ix+1, iy, iz+1] + dt * μ * ((Vz[ix+2, iy, iz+1] - Vz[ix, iy, iz+1])/(2*dx) + (Vx[ix+1, iy, iz+2] - Vx[ix+1, iy, iz])/(2*dz))
    end
    if (ix <= size(Syz, 1) && iy <= size(Syz, 2)-2 && iz <= size(Syz, 3)-2)
        Syz[ix, iy+1, iz+1] = damping_sxy * Syz[ix, iy+1, iz+1] + dt * μ * ((Vz[ix, iy+2, iz+1] - Vz[ix, iy, iz+1])/(2*dy) + (Vy[ix, iy+1, iz+2] - Vy[ix, iy+1, iz])/(2*dz))
    end
    return
end

function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end

##################################################
@views function elastic3D()
    # Physics
    lx, ly, lz = 60.0, 60.0, 60.0  # domain extends
    λ, μ       = 2.0, 1.0          # Lame parameters (λ, μ)
    k          = 1.0               # bulk modulus
    ρ          = 1.0               # density
    t          = 0.0               # physical time
    damping_sxx  = 0.80               # 1 - damping factor
    damping_sxy  = 0.85               # 1 - damping factor
    # Numerics
    nz         = 63
    nx, ny     = nz, nz   # numerical grid resolution
    nt         = 6000              # number of timesteps
    nout       = 40                # plotting frequency
    # Derived numerics
    me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz) # MPI initialisation
    select_device()                                               # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy, dz = lx/(nx_g()-1), ly/(ny_g()-1), lz/(nz_g()-1)      # cell sizes
    # Array allocations
    P          = @zeros(nx  ,ny  ,nz  )
    Vx         = @zeros(nx+1,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    Vz         = @zeros(nx  ,ny  ,nz+1)
    Sxx        = @zeros(nx  ,ny  ,nz  )
    Syy        = @zeros(nx  ,ny  ,nz  )
    Szz        = @zeros(nx  ,ny  ,nz  )
    Sxy        = @zeros(nx  ,ny  ,nz  )
    Sxz        = @zeros(nx  ,ny  ,nz  )
    Syz        = @zeros(nx  ,ny  ,nz  )
    # Initial conditions
    dt         = 0.5 * min(dx,dy,dz)/sqrt(k/ρ)/6.1 # prevent grad explosion
    # Initialize stress tensor components
    # Sxy .= Data.Array([exp(-(x_g(ix,dx,Sxy)-0.5*lx)^2 - (y_g(iy,dy,Sxy)-0.5*ly)^2 - (z_g(iz,dz,Sxy)-0.5*lz)^2) for ix=1:size(Sxy,1), iy=1:size(Sxy,2), iz=1:size(Sxy,3)])
    # Sxz .= Data.Array([exp(-(x_g(ix,dx,Sxz)-0.8*lx)^2 - (y_g(iy,dy,Sxz)-0.33*ly)^2 - (z_g(iz,dz,Sxz)-0.5*lz)^2) for ix=1:size(Sxz,1), iy=1:size(Sxz,2), iz=1:size(Sxz,3)])
    # Syz .= Data.Array([exp(-(x_g(ix,dx,Syz)-0.5*lx)^2 - (y_g(iy,dy,Syz)-0.5*ly)^2 - (z_g(iz,dz,Syz)-0.5*lz)^2) for ix=1:size(Syz,1), iy=1:size(Syz,2), iz=1:size(Syz,3)])
    # Initialize diagonal stress tensor components
    stress      = 500
    Sxx .= Data.Array([stress * exp(-(x_g(ix,dx,Sxx)-0.2*lx)^2 - (y_g(iy,dy,Sxx)-0.2*ly)^2 - (z_g(iz,dz,Sxx)-0*lz)^2) for ix=1:size(Sxx,1), iy=1:size(Sxx,2), iz=1:size(Sxx,3)])
    # Syy .= Data.Array([exp(-(x_g(ix,dx,Syy)-0.5*lx)^2 - (y_g(iy,dy,Syy)-0.5*ly)^2 - (z_g(iz,dz,Syy)-0.5*lz)^2) for ix=1:size(Syy,1), iy=1:size(Syy,2), iz=1:size(Syy,3)])
    # Szz .= Data.Array([exp(-(x_g(ix,dx,Szz)-0.5*lx)^2 - (y_g(iy,dy,Szz)-0.5*ly)^2 - (z_g(iz,dz,Szz)-0.5*lz)^2) for ix=1:size(Szz,1), iy=1:size(Szz,2), iz=1:size(Szz,3)])

    # Preparation of visualisation
    ENV["GKSwstype"]="nul"
    if (me==0)
        loadpath = "./viz3D6k_out-damp80m500s4/"
        if isdir(loadpath)==false mkdir(loadpath) end;  anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
    end
    nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
    if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
    P_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
    # Gather data for visualization
    Sxx_v = zeros(nx_v, ny_v, nz_v)
    Syy_v = zeros(nx_v, ny_v, nz_v)
    Szz_v = zeros(nx_v, ny_v, nz_v)
    Sxy_v = zeros(nx_v, ny_v, nz_v)
    Syz_v = zeros(nx_v, ny_v, nz_v)
    Sxz_v = zeros(nx_v, ny_v, nz_v)
    P_inn = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
    # Define inner arrays for stress tensor components
    Sxx_inn = zeros(nx-2, ny-2, nz-2)
    Syy_inn = zeros(nx-2, ny-2, nz-2)
    Szz_inn = zeros(nx-2, ny-2, nz-2)
    Sxy_inn = zeros(nx-2, ny-2, nz-2)
    Sxz_inn = zeros(nx-2, ny-2, nz-2)
    Syz_inn = zeros(nx-2, ny-2, nz-2)

    y_sl  = Int(ceil(ny_g()/2))
    Xi_g  = dx:dx:(lx-dx) # inner points only
    Zi_g  = dz:dz:(lz-dz)
    # Time loop
    iframe = 0
    for it = 1:nt
        if (it==11) tic() end
        @hide_communication (8, 8, 4) begin # communication/computation overlap
            @parallel compute_V!(Vx, Vy, Vz, Sxx,Syy,Szz,Sxy,Sxz,Syz, dt, ρ, dx, dy, dz)
            # update_halo!(Vx, Vy, Vz, Sxx,Syy,Szz,Sxy,Sxz,Syz)
            update_halo!(Vx, Vy, Vz)
        end
        @parallel compute_S!(Sxx,Syy,Szz,Sxy,Sxz,Syz, Vx, Vy, Vz, dt, λ, μ, dx, dy, dz, damping_sxx)
        @hide_communication (8, 8, 4) begin
            @parallel compute_S_diag!(Sxx,Syy,Szz,Sxy,Sxz,Syz, Vx, Vy, Vz, dt, λ, μ, dx, dy, dz, damping_sxy)
            update_halo!(Sxy,Sxz,Syz)
        end
        t = t + dt

        # Visualisation
        if mod(it,nout)==0
            # Assign the inner part of each global array to the corresponding inner array
            Sxx_inn .= Array(Sxx[2:end-1, 2:end-1, 2:end-1])
            Syy_inn .= Array(Syy[2:end-1, 2:end-1, 2:end-1])
            Szz_inn .= Array(Szz[2:end-1, 2:end-1, 2:end-1])
            Sxy_inn .= Array(Sxy[2:end-1, 2:end-1, 2:end-1])
            Sxz_inn .= Array(Sxz[2:end-1, 2:end-1, 2:end-1])
            Syz_inn .= Array(Syz[2:end-1, 2:end-1, 2:end-1])

            gather!(Sxx_inn, Sxx_v)
            gather!(Syy_inn, Syy_v)
            gather!(Szz_inn, Szz_v)
            gather!(Sxy_inn, Sxy_v)
            gather!(Syz_inn, Syz_v)
            gather!(Sxz_inn, Sxz_v)

            # Compute von Mises stress
            # von_Mises_stress = sqrt.(0.5 .* ((Sxx_v - Syy_v).^2 + (Syy_v - Szz_v).^2 + (Szz_v - Sxx_v).^2) .+ 6 .* (Sxy_v.^2 + Syz_v.^2 + Sxz_v.^2))
            # Compute sum of squared stresses
            sum_squared_stresses = Sxx_v.^2 + Syy_v.^2 + Szz_v.^2 + Sxy_v.^2 + Syz_v.^2 + Sxz_v.^2
            # stress = Sxx_v 
            stress = sum_squared_stresses

            # Plot von Mises stress
            if me == 0
                # heatmap(Xi_g, Zi_g, von_Mises_stress[:, y_sl, :]', aspect_ratio=1, c=:viridis, title="von Mises Stress at t=$t")
                print(Sxx_v[60,60,30:40])
                print("\n")
                print(stress[60,60,30:40])
                print("\n")
                if iframe < 150
                    heatmap(Xi_g, Zi_g, stress[:, y_sl, :]', aspect_ratio=1, c=:viridis, title="Sum of Squared Stresses at t=$t")
                    frame(anim)
                end
                if it == nout
                    print(size(stress,1),size(stress,2),size(stress,3))
                end
                save_array(@sprintf("%sout_T_%04d", loadpath, iframe+=1), convert.(Float32, Array(stress)))

            end
        end
    end
    # Performance
    wtime    = toc()
    A_eff    = (4*2)/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                           # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                          # Effective memory throughput [GB/s]
    if (me==0) @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=2)) end
    # if (me==0) gif(anim, "wave3D.gif", fps = 15) end
    finalize_global_grid()
    return

end

elastic3D()
