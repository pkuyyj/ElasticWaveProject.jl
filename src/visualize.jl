using GLMakie, Plots,Printf

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

function visualise!(str,i,anim,lx=120.0, nt=10000, nout=200)
    ly, lz = 40.0, 20.0, 20.0
    nz          = 63
    # nx,ny       = 2 * (nz + 1) - 1, nz
    nx,ny,nz    = 122,122,61

    T  = zeros(Float32, nx,ny,nz)
    load_array(str, T)
    print(T[5,5,10:20])
    log_T = log10.(T .+ eps())
    dx,dy,dz    = lx/nx, ly/ny, lz/nz
    # xc, yc, zc = LinRange(-lx/2, lx/2, nx)[2:end-1], LinRange(-ly/2, ly/2, ny)[2:end-1], LinRange(0, lz, nz)[2:end-1]
    xc, yc, zc = LinRange(0+ dx + dx / 2, lx  - dx - dx / 2, nx),LinRange(-ly / 2 + dy + dy / 2, ly / 2 - dy - dy / 2, ny), LinRange(-lz + dz + dz / 2, -dz - dz / 2, nz) # inner points only

    fig = Figure(resolution=(1600, 1000), fontsize=24)
    ax  = Axis3(fig[1, 1]; aspect=(1, 1, 0.5), title="Stress", xlabel="lx", ylabel="ly", zlabel="lz")
    cmax = 1.0
    cmin = 0.0
    surf_T = contour!(ax, xc, yc, zc, log_T; alpha=0.1, colormap=:turbo)
    # surf_T = contour!(ax, xc, yc, zc, log_T; alpha=0.1, clims=(cmin,cmax))

    
    save(@sprintf("./viz3D_out/stress_3D_%04d.png",i), fig)
    return fig
end

anim = Animation("./viz3D_out",String[])
for i = 1:150
    str = @sprintf("./viz3D_out/out_Stress_%04d", i)
    fig = visualise!(str,i,anim)
    # frame(anim)
end

# gif(anim, "./docs/wave6k.gif", fps = 20)