import Pkg; Pkg.add("Literate"); Pkg.instantiate()
using Literate

# directory where the markdown files are put
md_dir = joinpath(@__DIR__, "../docs")
Literate.markdown("wave3D_multixpu.jl", md_dir, execute=false, documenter=false, credit=false)
