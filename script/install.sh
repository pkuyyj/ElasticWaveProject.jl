module load daint-gpu
module load Julia/1.9.3-CrayGNU-21.09-cuda
export MPICH_RDMA_ENABLED_CUDA=1
export IGG_CUDAAWARE_MPI=1
julia --project src/install.jl 