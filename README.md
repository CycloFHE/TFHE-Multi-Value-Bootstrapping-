# Companion code

Julia programs for *Rotate Once, Read Many Times: on the Output
Noise of Multi-Value Bootstrapping*.

`cofactor_svp.jl` solves stage (a) of Section 5.1 exactly: the shortest-vector problem
that selects the optimal cofactor. `mvm_modes.jl` implements the three bootstrapping
modes of the paper and produces the measurements of Section 6. Neither includes the
other, and neither reads anything from the rest of the project.

## Requirements

`cofactor_svp.jl` needs one package:

```julia
import Pkg; Pkg.add("Hecke")
```

`mvm_modes.jl` runs on the standard library alone. Two packages are used when present: 
**FFTW**, which is what makes ring degrees in the thousands reachable, and 
**Plots**, without which the numbers are written as CSV instead of being drawn.

```julia
import Pkg; Pkg.add(["FFTW", "Plots"])
```

## cofactor_svp.jl — the optimal cofactor at a random lift

The program builds the Gram matrix of Lemma 3,

```
G[k,k'] = p m^2 ( (p^2-1)/12 - c(p-c)/2 ),   c = (k-k') mod p,   m = p^(alpha-1),
```

which is integral and positive definite of size `p-1`. Because `G` is
integral, Hecke's lattice enumeration solves it exactly, in integer arithmetic.

The two designs of Corollary 1 (`gamma = 1`
and `gamma = M Omega*_0`). The program checks, prime by prime,
that no other cofactor beats the better of the two.

```
julia cofactor_svp.jl            # sweeps every odd prime p <= 97 at alpha = 1
julia cofactor_svp.jl 13         # one setting: p = 13, alpha = 1
julia cofactor_svp.jl 13 2       # one setting: p = 13, alpha = 2
```

In the REPL, `include("cofactor_svp.jl")` gives the pieces separately:
`gram_G(p, alpha)` for the matrix, `cost(p, alpha, g)` for the exact rational cost of a
cofactor given by its `N` coordinates, `gamma_one` and `gamma_pivot` for the two
designs, `stage_a(p, alpha)` for the enumeration, and `sweep(primes; alpha)` for the
loop.

## mvm_modes.jl — the three modes and the measurements of Section 6

One file, three bootstrapping modes:

`SVM` is the single-value mode, which rotates `v_f = (Omega*_0)^{-1} v*_f` and reads the
accumulator through `U* = Omega*_0`. `MVM1` is the multi-value mode at the canonical
cofactor `gamma = 1`, and `MVMpiv` the multi-value mode at the pivot
`gamma = M Omega*_0`. All three are coded in the primal, on the monomial basis: the
accumulator is an element of `T = K/R`, and multiplying the final accumulator by
`Omega*_0` turns it into an RLWE*, which is how its error is inspected.

Every RLWE row is drawn as an RLWE* row, with error spherical in the canonical embedding. 
The torus is carried in `Float64`.

### Self-tests

```
julia -t auto mvm_modes.jl
```

runs the deterministic checks at `(p, alpha) = (7,1), (11,1), (101,1), (11,2)`: the ring
product against the canonical embeddings (R0), the membership statements
`v_f in (1/p)R` and `w(pivot) in (1/p^2)R` (R1, R2), the factorisation
`v*_f = V*_f . w` (R3), the invariance of `B*_f . w` under the choice of cofactor (R4),
the batched external product (R5), the identity `E = Tr(U* e)` of Proposition 3, 
and the gadget-decomposition bounds (T1a–T1c). 

### Experiments

```
julia -t auto mvm_modes.jl run         # full sample
julia -t auto mvm_modes.jl run quick   # the same at a reduced sample size
```

### From the REPL

The file ends with the usual `PROGRAM_FILE` guard, so `include` loads it without running
anything:

```julia
julia> include("mvm_modes.jl")
julia> Threads.nthreads(), HAVE_FFTW, HAVE_PLOTS   # what was picked up

julia> selftest(7, 1)
julia> amp = exp_amplification(7, 2; nb = 3000, n = 512, seed = 11)
julia> cov = [exp_covariance(p, a; nb = 6000, n = 128, seed = 21)
              for (p, a) in ((7,1), (11,1), (17,1), (5,2), (7,2), (3,4))]
julia> println(table1(cov))          # the table as text
julia> print(table1_tex(cov))        # the same in LaTeX
julia> figures(amp, cov; prefix = "trial")
julia> main(["run", "quick"])        # or simply replay the script
```

Two default sample sizes differ between the entry points: `exp_amplification` takes
`nb = 8000` and `exp_covariance` takes `nb = 3000`, whereas the `run` command passes
8000 and 6000 respectively. Pass `nb` explicitly to reproduce a published setting.

### Reproducing the paper, in one call

```julia
julia> run_paper()
```

It covers the five amplification settings
`(p, alpha) = (5,5), (7,4), (11,3), (13,3), (37,2)`, of ring degrees
`N = 2500, 2058, 1210, 2028, 1332`, and the six covariance settings
`(7,1), (11,1), (17,1), (5,2), (7,2), (3,4)`, under both noise models.

The covariance settings are deliberately small, `N <= 54`. What is measured there is the
full `N x N` covariance of `e*`, that is `N(N+1)/2` free parameters, which no feasible
sample estimates at `N = 2058`.

## Where the output goes

`run` and the plotting helpers use relative names, so their output lands in the **current
directory**: `mvm_jl_amp.csv`, `mvm_jl_cov.csv`, `mvm_jl_gram.csv`,
`mvm_jl_table1.tex`, and, when Plots is installed, `mvm_jl_funcs`, `mvm_jl_cov` and
`mvm_jl_cloud` as both `.pdf` and `.png`.

`run_paper` is different: its `prefix` defaults to `joinpath(@__DIR__, "mvm_paper")`, so
its output always lands **next to this file**.

## A note on numbering

Read "Section 5" for the numerical validation, now Section 6, and "Table 1" for the
covariance table, now Table 2. The mathematical references — Lemma 3, Proposition 3,
Corollary 1, Assumption 1 — match the paper as it stands.
