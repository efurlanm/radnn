# SDumont Manual - Singularity

Notes on Singularity, available in the SDumont manual: <https://github.com/lncc-sered/manual-sdumont/wiki/>


This part is roughly standard in slurm batch script.

```bash
#SBATCH --nodes=4 
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=4
#SBATCH -p sequana_gpu_dev
#SBATCH -J XXXX
```


Runs on a single node:

```
export GMX_ENABLE_DIRECT_GPU_COMM=1

export SINGULARITY="singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd gromacs_2020.2.sif"

${SINGULARITY} gmx mdrun -ntmpi $SLURM_NTASKS -ntomp $SLURM_CPUS_PER_TASK  -v -s topol.tpr -deffnm stmv  -nsteps 100000 -resetstep 90000 -noconfout -dlb no -nstlist 300

```

Uses multiple nodes:

```
export GMX_ENABLE_DIRECT_GPU_COMM=1

export SINGULARITY="singularity run --nv gromacs_2020.6.sif"

mpirun ${SINGULARITY} gmx mdrun -ntomp $SLURM_CPUS_PER_TASK  -v -s topol.tpr -deffnm stmv  -nsteps 100000 -resetstep 90000 -noconfout -dlb no -nstlist 300
```

https://www.lammps.org/ :

```
export SINGULARITY="singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd lammps_patch_3Nov2022.sif"

srun --mpi=pmi2 $SINGULARITY lmp -k on g 4 -sf kk -pk kokkos cuda/aware on neigh full comm device binsize 2.8 -in  in.spce.ehex
```

https://hub.docker.com/r/lammps/lammps/tags

```
export SINGULARITY="singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd lammps_patch_7Jan2022_rockylinux8_openmpi_py3.sif"

srun --mpi=pmix_v3 $SINGULARITY lmp -in  in.spce.ehex
```

https://catalog.ngc.nvidia.com/orgs/hpc/containers/namd

```
export NAMD_EXE=namd3
export SINGULARITY="singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd namd_3.0-beta5.sif"

$SINGULARITY $NAMD_EXE  +p${SLURM_NTASKS} +devices 0,1,2,3 +setcpuaffinity apoa1_nve_cuda_soa.namd
```

https://www.cp2k.org/

```
export SINGULARITY="singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd cp2k_v9.1.0.sif"

srun --mpi=pmi2  $SINGULARITY cp2k.psmp -i H2O-dft-ls.NREP2.inp
```

https://catalog.ngc.nvidia.com/orgs/hpc/containers/cp2k

```
export SINGULARITY="singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd cp2k_2022.1.sif"

srun --mpi=pmi2  $SINGULARITY cp2k.psmp -i H2O-dft-ls.NREP2.inp
```

https://catalog.ngc.nvidia.com/orgs/hpc/containers/quantum_espresso

```
export SINGULARITY="singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd quantum_espresso_qe-7.3.1.sif"

srun --mpi=pmix_v3  $SINGULARITY pw.x -input ausurf.in
```
