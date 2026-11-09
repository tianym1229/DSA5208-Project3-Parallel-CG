# DSA5208 Project 3

## Communication-Aware Parallel Conjugate Gradient for Large Sparse Linear Systems

**Platform:** NUS Vanda HPC\
**Implementations:** Serial C++, MPI, MPI + OpenMP, and CUDA C++\
**Primary problem:** Synthetic 2D and 3D sparse symmetric
positive-definite (SPD) Laplacian systems

------------------------------------------------------------------------

## 1. Project overview

This project implements and evaluates the Conjugate Gradient (CG) method
for solving sparse linear systems

$$
Ax=b
$$

where $A$ is symmetric positive definite. The work progresses from a
serial C++ implementation to distributed-memory MPI, hybrid MPI +
OpenMP, and a separate CUDA implementation for an NVIDIA GPU.

The project investigates:

-   numerical correctness and validation;
-   serial runtime growth as problem size and matrix nonzero count
    increase;
-   MPI strong and weak scaling up to 256 MPI ranks;
-   communication costs, including halo exchange and global reductions;
-   MPI-rank/OpenMP-thread configurations;
-   GPU correctness, solver timing, and CPU/GPU comparison.

The archived CSV files and PNG figures are the evidence from the
completed experiments. The PBS files and test programs are supplied for
building, checking, and conducting future runs. A future run creates new
measurements; it does not recreate the historical job IDs or guarantee
identical timings.

## 2. Mathematical problem and validation

### 2.1 Conjugate Gradient

CG is an iterative method for SPD systems. At iteration $k$, it updates
the solution estimate $x_k$, residual $r_k$, and search direction $p_k$:

$$
\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}, \qquad x_{k+1}=x_k+\alpha_k p_k,
$$

$$
r_{k+1}=r_k-\alpha_k A p_k, \qquad \beta_k=\frac{r_{k+1}^T r_{k+1}}{r_k^T r_k},
$$

$$
p_{k+1}=r_{k+1}+\beta_k p_k.
$$

The main computational operations are sparse matrix-vector
multiplication (SpMV), vector dot products, vector updates, and
convergence checks.

### 2.2 Synthetic 2D and 3D systems

The experiments use structured-grid discrete Laplacian matrices:

-   **2D:** five-point stencil;
-   **3D:** seven-point stencil.

The sparse matrices are stored in Compressed Sparse Row (CSR) format.
CSR stores nonzero values, their column indices, and row offsets.

A manufactured-solution approach supplies a known reference vector
$x_{\mathrm{true}}$, then constructs

$$
b=Ax_{\mathrm{true}}.
$$

This allows the computed solution to be checked against a known answer.
The principal validation quantities are

$$
\text{Relative Residual}=\frac{\|Ax-b\|_2}{\|b\|_2},
\qquad
\text{Relative Error}=\frac{\|x-x_{\mathrm{true}}\|_2}{\|x_{\mathrm{true}}\|_2}.
$$

A residual tolerance of $10^{-8}$ is used in the recorded
benchmark/correctness workflows described by the corresponding scripts.
Check each test script for its exact additional assertions; not every
test uses identical criteria.

### 2.3 Why parallel CG requires communication

CG iterations are dependent: an iteration needs quantities computed in
the preceding iteration. With MPI domain decomposition, a rank may need
vector entries owned by neighbouring ranks for its local SpMV. These
boundary values are exchanged through a **halo exchange**. Dot products
that combine contributions from all ranks require a collective reduction
such as `MPI_Allreduce`.

The communication and synchronization costs can therefore limit scaling,
even when the local arithmetic is parallelized.

## 3. Project requirements and experimental coverage

The following table maps the project work to the broad components in the
course project guidance.

  -----------------------------------------------------------------------
  Course component        Project work            Archived evidence
  ----------------------- ----------------------- -----------------------
  Algorithm explanation   CG, SPD Laplacian       Source code, this
                          systems, CSR,           README, final report
                          manufactured solution,  
                          validation              

  Serial runtime scaling  Five 2D and five 3D     `data/serial/`,
                          grid configurations;    `figures/serial/`
                          five formal repeats per 
                          configuration           

  MPI performance         Strong and weak         `data/mpi/`,
                          scaling, up to 256      `figures/mpi/`
                          ranks; speedup,         
                          efficiency,             
                          communication analysis  

  MPI + X                 MPI + OpenMP            `data/hybrid/`,
                          rank/thread             `figures/hybrid/`
                          configurations and      
                          comparisons             

  GPU test for the OpenMP Separate CUDA CG        `data/gpu/`,
  route                   correctness and         `figures/gpu/`
                          benchmark on an NVIDIA  
                          A40                     

  Overall discussion      Compare execution       Final project report
                          models, scaling limits, 
                          communication, and      
                          design choices          
  -----------------------------------------------------------------------

The course guidance also asks the report to define non-standard
terminology, explain synthetic data generation and validation, and
include informative plots. Axes representing time, problem size, or CPU
count should use logarithmic scales where applicable. The final report
should explicitly address these points rather than relying on the README
alone.

## 4. Experimental stages and recorded scope

### Stage A --- Algorithm and correctness foundation

-   Define 2D and 3D sparse Laplacian systems.
-   Construct CSR matrices.
-   Generate a known solution and form $b=Ax_{\mathrm{true}}$.
-   Validate convergence using residual and solution error.

### Stage B --- Serial C++ baseline

The archived serial study covers ten configurations:

  Dimension   Grid sizes
  ----------- ------------------------
  2D          64, 128, 256, 512, 768
  3D          16, 32, 40, 64, 80

There are five formal observations per configuration (50 observations in
total). The analysis considers runtime, iteration count, time per
iteration, variability, and empirical power-law fits. Matrix
dimension/problem size and NNZ should both be discussed in the report.

The serial audit identified high variability in the formal measurements
for 2D Grid 512 and 3D Grid 64. Independent recheck measurements were
collected for those configurations. The report should disclose this and
explain which dataset is used for each reported summary; do not silently
replace or discard the original observations.

### Stage C5 --- MPI distributed-memory scaling

**Strong scaling** holds the global problem fixed while increasing the
number of MPI ranks. The archived campaign includes 2D and 3D cases,
communication modes, and rank counts up to 256. Report speedup and
efficiency with the reference configuration stated explicitly.

**Weak scaling** increases the global problem as the rank count
increases, using the experiment's rank-to-grid mapping to keep the
per-rank workload approximately controlled. The report should state that
mapping rather than describing it only as a general intention.

The communication analysis includes halo exchange, `MPI_Allreduce`, and
baseline-versus-overlap comparisons. Communication timing categories may
overlap or include waiting; do not automatically add them as if they
were mutually exclusive components of total runtime.

### Stage C6 --- MPI + OpenMP hybrid parallelism

The formal hybrid study covers 4, 8, 16, and 32 CPU cores, with
different MPI-rank/OpenMP-thread decompositions, 2D and 3D cases, and
baseline/overlap modes. The archived campaign records 48 configurations
and 240 formal measurements (five formal repeats per configuration, in
addition to warm-up runs).

**Comparison definition:** the MPI-only reference in the hybrid analysis
uses one OpenMP thread per MPI rank in the OpenMP-enabled executable. It
is not a separately compiled MPI-only executable. Hybrid speedup and
efficiency use the four-core reference; do not confuse this with the
one-rank reference used for MPI strong scaling.

The results show that hybrid parallelism is not uniformly faster. The
report should explain the differences by dimension, core count, and
rank/thread decomposition.

### Stage C7 --- CUDA GPU implementation

The GPU implementation is a separate CUDA solver, not an MPI + CUDA
multi-GPU implementation. The recorded environment uses an NVIDIA A40
and CUDA 12.4. The implementation uses double-precision arithmetic and 
CSR storage, with GPU-accelerated routines for sparse matrix-vector 
multiplication and dense-vector operations. The CG iteration logic is 
managed by the host program, while computationally intensive operations 
are executed on the GPU.

Recorded correctness coverage: four configurations, with all four
passing in the supplied test output.

Formal GPU benchmark configurations:

-   2D Grid 512;
-   3D Grid 64;
-   262,144 unknowns in each configuration;
-   one warm-up and five formal measurements per configuration.

CPU/GPU comparisons in the archived analysis are **solver-time ratios**,
not end-to-end application speedups. Timing boundaries and the high
variability observed in some historical CPU measurements must be
acknowledged when interpreting the ratios.

## 5. Main findings and interpretation

The archived analyses support the following high-level observations.
They apply to the tested implementations, problem sizes, and Vanda
environment; they should not be generalized to every sparse system or
hardware platform.

1.  **Serial:** Runtime growth reflects both changes in iteration count
    and the cost of each iteration. Empirical power-law fits describe
    the measured finite range; they do not prove an asymptotic
    complexity result.
2.  **MPI strong scaling:** For the tested baseline cases, the median
    runtime reached its minimum at 16 ranks and increased at higher rank
    counts. Communication and synchronization become increasingly
    important.
3.  **MPI weak scaling:** End-to-end weak-scaling efficiency declined as
    rank count and global problem size increased. Iteration-count growth
    contributes to the observed trend.
4.  **MPI overlap:** Nonblocking communication/overlap did not
    automatically improve runtime; the measured result depends on
    whether useful work can hide communication.
5.  **MPI + OpenMP:** Hybrid decomposition was not uniformly better than
    one thread per MPI rank. Performance depended on problem dimension
    and core count.
6.  **CUDA:** The GPU correctness tests passed and formal solver timings
    were recorded on the A40. CPU/GPU solver-time ratios require the
    timing-scope and CPU-variability caveats described above.

The final report should develop these observations into a connected
discussion: explain why the results occurred, identify the
communication-heavy operations, and distinguish measured evidence from
interpretation.

## 6. Repository layout

``` text
DSA5208_Project3_Submission/
├── Makefile
├── README.md
├── include/
│   └── cg.hpp
├── src/
│   ├── main.cpp
│   ├── cg.cpp
│   ├── mpi_main.cpp
│   └── gpu/
│       └── gpu_main.cu
├── tests/
│   ├── test_serial.py
│   ├── test_mpi.py
│   ├── test_hybrid.py
│   ├── test_benchmark.py
│   ├── test_part2.py
│   └── gpu/
├── pbs/
│   ├── serial/
│   ├── mpi/
│   ├── hybrid/
│   └── gpu/
├── analysis/
│   ├── serial/
│   ├── mpi/
│   │   └── legacy_step5/
│   ├── hybrid/
│   └── gpu/
├── data/
│   ├── serial/
│   ├── mpi/
│   │   ├── strong_scaling/
│   │   └── weak_scaling/
│   ├── hybrid/
│   └── gpu/
├── figures/
│   ├── serial/
│   ├── mpi/
│   │   ├── strong_scaling/
│   │   └── weak_scaling/
│   ├── hybrid/
│   └── gpu/
└── results/
    ├── analysis/
    └── formal/
```

### Directory responsibilities

  -----------------------------------------------------------------------
  Directory                           Contents
  ----------------------------------- -----------------------------------
  `src/`, `include/`                  C++ and CUDA implementation source
                                      and header

  `tests/`                            Correctness, benchmark, and
                                      comparison test programs

  `pbs/`                              Vanda PBS job scripts for smoke
                                      tests and formal runs

  `analysis/`                         Scripts for statistical analysis
                                      and figure generation

  `data/`                             Archived CSV inputs, summaries, and
                                      integrity manifests

  `figures/`                          Selected final PNG figures; no PDF
                                      figures are required in this
                                      package

  `results/analysis/`                 Outputs generated by analysis and
                                      smoke-test workflows

  `results/formal/`                   Outputs from future formal PBS runs
  -----------------------------------------------------------------------

The `analysis/mpi/legacy_step5/` directory preserves earlier MPI
analysis scripts for historical reference. These scripts use legacy
input/output paths and are not part of the active final analysis
workflow.

## 7. Software and environment

The project was developed and tested on NUS Vanda HPC.

  Component                   Recorded environment / requirement
  --------------------------- ---------------------------------------------------
  C++                         C++17 compiler (`g++`)
  MPI                         Open MPI 4.1.8 (`mpicxx`, `mpirun`)
  Shared-memory parallelism   OpenMP
  GPU                         NVIDIA GPU; recorded formal tests used A40
  CUDA                        CUDA Toolkit 12.4
  GPU libraries               cuBLAS and cuSPARSE
  Python                      Python 3 with standard library for test harnesses
  Plotting                    Matplotlib for analysis/figure-generation scripts

Load the relevant modules on Vanda before compiling or submitting the
corresponding jobs. MPI and CUDA modules are not required for a
serial-only build. GPU compilation and execution require a suitable CUDA
environment and GPU allocation.

## 8. Build instructions

Run commands from the project root.

### 8.1 Build CPU executables

``` bash
module load openmpi/gcc/64/4.1.8
make -j1 all
```

This builds:

-   `cg_serial`
-   `cg_mpi`
-   `cg_hybrid`

### 8.2 Build the GPU executable

``` bash
module load CUDA/12.4.0
make -j1 gpu
```

The GPU executable is created at:

``` text
build/cg_gpu
```

### 8.3 Clean generated executables

``` bash
make clean
```

`make clean` removes the generated CPU executables and `build/cg_gpu`.
It does not remove archived CSV data or PNG figures.

## 9. Correctness tests and PBS execution

### 9.1 Serial test

After building the serial executable:

``` bash
make serial-test
```

### 9.2 MPI test

The MPI test requires an MPI allocation with sufficient slots. Submit
the supplied smoke-test job from the project root:

``` bash
qsub pbs/mpi/mpi_smoke.pbs
```

### 9.3 Hybrid test

The hybrid test requires an appropriate MPI/OpenMP allocation:

``` bash
qsub pbs/hybrid/hybrid_smoke.pbs
```

### 9.4 GPU correctness smoke test

The GPU smoke test builds the GPU executable and runs the correctness
harness on a GPU allocation:

``` bash
qsub pbs/gpu/gpu_smoke.pbs
```

### 9.5 Recorded test evidence

The supplied terminal records show:

  -----------------------------------------------------------------------
  Test / check                                            Recorded result
  ------------------------------ ----------------------------------------
  Serial correctness smoke tests                                12 passed

  MPI correctness smoke tests                                   30 passed

  Hybrid MPI/OpenMP comparisons                                 32 passed

  GPU correctness tests                                        4/4 passed

  CPU clean-build executables               Serial, MPI, and Hybrid built
                                                             successfully

  GPU clean build                     Built successfully; cuBLAS/cuSPARSE
                                                    dependencies resolved
  -----------------------------------------------------------------------

These are results from the recorded runs, not a claim that every test
has just been rerun from a fresh checkout. The MPI and hybrid test
programs are designed for PBS allocations; do not assume they can be run
directly on a login node.

## 10. Analysis and figure-generation scripts

The active analysis scripts are grouped by implementation:

  ----------------------------------------------------------------------------------
  Area                                Main scripts
  ----------------------------------- ----------------------------------------------
  Serial                              `analysis/serial/analyze_serial.py`,
                                      `audit_serial.py`, `analyze_complexity.py`,
                                      `plot_complexity.py`, `plot_serial_final.py`

  MPI                                 `analysis/mpi/plot_strong_scaling_final.py`,
                                      `plot_stepC5_weak_scaling.py`

  Hybrid                              `analysis/hybrid/analyze_hybrid_formal.py`,
                                      `analyze_hybrid_comparison.py`,
                                      `analyze_hybrid_scaling.py`, and the
                                      H4/H5/H6/final plotting scripts

  GPU                                 `analysis/gpu/compare_cpu_gpu.py`,
                                      `plot_gpu_performance.py`,
                                      `plot_gpu_components_v4_logbar.py`
  ----------------------------------------------------------------------------------

The active scripts use project-relative data and output locations where
applicable. Some analysis scripts consume intermediate files produced by
another script; follow the input paths in each script and run the
upstream analysis first.

**Important reproducibility note:** the archived PNG figures in
`figures/` are the selected figures for review and reporting. Figure
regeneration is optional when the purpose is only to inspect or submit
the archived results. It requires the Python plotting dependencies
(including Matplotlib) and correctly prepared input CSV files. A Python
syntax check does not prove that all runtime data dependencies are
present or that every plotting pipeline has been executed successfully.

The legacy scripts under `analysis/mpi/legacy_step5/` are retained for
reference and should not be treated as the active MPI analysis pipeline.

## 11. Data and figure inventory

The submission package contains archived CSV datasets and selected PNG
figures, grouped by experiment:

-   `data/serial/` and `figures/serial/`
-   `data/mpi/strong_scaling/` and `figures/mpi/strong_scaling/`
-   `data/mpi/weak_scaling/` and `figures/mpi/weak_scaling/`
-   `data/hybrid/` and `figures/hybrid/`
-   `data/gpu/` and `figures/gpu/`

The package inventory recorded during assembly contained 28 CSV files
and 28 PNG figures. The MPI strong-scaling final figure directory
contains six selected PNGs. PDF figures are intentionally excluded.

CSV files are the numerical evidence; PNG files are the selected visual
summaries. Do not substitute terminal logs for figures in the final
report. The course guidance explicitly says not to include raw log files
or terminal output in the final submission.

## 12. PBS scripts and future formal runs

PBS scripts are grouped by execution model:

-   `pbs/serial/`
-   `pbs/mpi/`
-   `pbs/hybrid/`
-   `pbs/gpu/`

The smoke-test scripts build and test the corresponding implementation.
The formal scripts are retained as reproducibility templates and write
new measurements under `results/formal/`.

Before submitting a formal job:

1.  Confirm that the correct modules are loaded by the script.
2.  Confirm that the requested resources match the intended experiment.
3.  Confirm that the executable is built from the current source.
4.  Check the output path and ensure that the job will not overwrite
    archived evidence.
5.  Inspect the PBS output and generated CSV after completion.

The updated PBS scripts passed shell syntax checks in the recorded
audit. Static syntax validation is not the same as a successful new
formal run. Historical experiment CSVs under `data/` should remain
unchanged.

## 13. Performance metric definitions

Use consistent definitions throughout the report.

-   **Runtime:** elapsed time for the timing region defined by the
    relevant implementation.
-   **Speedup:** reference runtime divided by parallel runtime. Always
    identify the reference.
-   **Parallel efficiency:** speedup divided by the corresponding
    processor/core-count increase, using the stated reference.
-   **Strong scaling:** fixed global problem size while increasing the
    number of processors.
-   **Weak scaling:** increase problem size with processor count while
    keeping per-process workload approximately controlled.
-   **NNZ:** number of nonzero matrix entries.
-   **SpMV:** sparse matrix-vector multiplication.
-   **Halo exchange:** transfer of boundary vector values required by
    neighbouring subdomains.
-   **Allreduce:** collective MPI operation that combines values across
    ranks.
-   **Overlap:** attempt to perform useful computation while
    communication is in progress.
-   **H2D / D2H:** host-to-device / device-to-host data transfer.
-   **CV:** coefficient of variation, calculated as sample standard
    deviation divided by the mean, expressed as a percentage.

The project uses different references for different comparisons: MPI
strong scaling uses a one-rank reference; hybrid scaling uses a
four-core reference; GPU comparison reports CPU/GPU solver-time ratios.
These quantities are not interchangeable.

## 14. Final report and submission checklist

The README is a guide to the code and archived evidence; it is not a
replacement for the assessed report.

The final report should include:

-   [ ] CG algorithm and why the problem satisfies the SPD requirement.
-   [ ] 2D/3D synthetic problem construction and manufactured-solution
    validation.
-   [ ] Definitions of non-standard terms and performance metrics.
-   [ ] Serial scaling with both matrix dimension/problem size and NNZ.
-   [ ] MPI strong and weak scaling through at least 256 CPU cores.
-   [ ] Parallel speedup, efficiency, and communication impact.
-   [ ] MPI + OpenMP decomposition and comparison methodology.
-   [ ] At least one GPU test for the OpenMP project route.
-   [ ] Explanation of the principal findings, limitations, and design
    choices.
-   [ ] Informative figures with appropriate logarithmic axes for time,
    problem size, or CPU count.
-   [ ] AI-use declaration in the format required by the course.

The final code package should contain source code, required plots,
relevant data, analysis/test scripts, PBS scripts, and this README. Do
not include raw terminal logs or job output files in the final
submission unless the instructor explicitly requests them.

## 15. Interpretation limits

The results are specific to the tested CG implementation, synthetic
Laplacian systems, tested problem sizes, and the Vanda hardware/software
environment. Empirical scaling fits are descriptive over the measured
range. CPU/GPU timing ratios must be interpreted using their timing
boundaries. Communication timing categories may overlap. These
qualifications should be retained in the final report.

------------------------------------------------------------------------

**End of README**
