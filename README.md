Optimal Factorization for Multi-Value Bootstrapping in TFHE/FHEW

This repository contains the Julia implementation accompanying the paper:

"Optimal Factorization for Multi-Value Bootstrapping in TFHE/FHEW"

by Philippe Chartier and Mohammed Lemou.

The paper introduces an optimal multi-value polynomial decomposition enabling an efficient multi-value bootstrapping procedure for TFHE/FHEW homomorphic encryption schemes.

This repository provides implementations of both the standard bootstrapping approach and the proposed multi-value method, together with benchmarking tools used to evaluate and compare their performance.

The associated paper will be made available on the IACR Cryptology ePrint Archive.

Repository Structure
.
├── standboot.jl
├── multival_boot.jl
├── bench_lut_amplification.jl
└── README.md

Repository Contents
standboot.jl

Reference implementation of the standard TFHE/FHEW bootstrapping algorithm used as the baseline method.

multival_boot.jl

Implementation of the proposed multi-value bootstrapping algorithm based on the optimal multi-value polynomial decomposition introduced in the paper.

bench_lut_amplification.jl

Benchmark script comparing standard and multi-value bootstrapping. It evaluates the impact of the proposed factorization on LUT amplification and reproduces the experimental results discussed in the paper.

Requirements
Julia 1.11 or later
Standard libraries only:
Random
Statistics
Printf
Base.Threads

No external Julia packages are required.

Running the Experiments

Run the scripts from the repository root:

julia standboot.jl
julia multival_boot.jl
julia bench_lut_amplification.jl


The benchmark can take advantage of Julia multithreading:

julia -t auto bench_lut_amplification.jl


or with a fixed number of threads:

julia -t 8 bench_lut_amplification.jl

Purpose

This repository aims to:

provide a reference implementation of standard TFHE/FHEW bootstrapping;
implement the proposed optimal multi-value polynomial decomposition;
demonstrate the resulting multi-value bootstrapping procedure;
reproduce the experimental evaluation presented in the accompanying paper.
Reproducibility

All experiments are implemented using Julia 1.11 and rely only on Julia standard libraries.

The script bench_lut_amplification.jl provides the benchmark framework used to compare the standard bootstrapping algorithm with the proposed multi-value approach.

Citation

If you use this repository in your research, please cite:

@misc{chartier2026optimal,
      title={Optimal Factorization for Multi-Value Bootstrapping in TFHE/FHEW},
      author={Philippe Chartier and Mohammed Lemou},
      year={2026},
      note={Available on the IACR Cryptology ePrint Archive}
}

Authors
Philippe Chartier
Mohammed Lemou

License

This repository is distributed under the terms of the license provided in the LICENSE file.
