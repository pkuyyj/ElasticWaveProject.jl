import Pkg; Pkg.precompile()
using Pkg
Pkg.instantiate()
using MPI
MPI.install_mpiexecjl()