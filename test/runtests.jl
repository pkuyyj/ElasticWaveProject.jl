using Test
using ElasticWaveProject

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()

    printstyled("Testing Elastic Wave 3D.jl\n"; bold=true, color=:white)

    run(`$exename -O3 --startup-file=no $(joinpath(testdir, "test3D.jl"))`)

    return 0
end

exit(runtests())