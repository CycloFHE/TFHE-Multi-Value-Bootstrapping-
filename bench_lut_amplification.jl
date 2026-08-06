#=============================================================================
 bench_lut_amplification.jl  --  Section 6 (Optim_LUT): illustrates points (ii)
 and (iii) of Proposition prop:facto at a real 128-bit-secure set (p=11, alpha=3,
 N=1210, sig_BK = sig_min(N)).

 For a range of functions f : T_p -> T_p (named + random admissible tables) it
 measures the two output-error std's of Proposition estimation1-erreur:
     sigma(E1)  (SVM / standboot: rotate v_f*, extract),
     sigma(E2)  (MVM / lutboot:   rotate w, apply V_f*, extract),
 and records, per function:
     ||V_f*||_Tr      (trace norm, the optimality functional of point (ii)),
     newfactor = sqrt( (sum_{i=0}^{p-1} F_i^2) / 2 )   (the amplification of
                 point (iii) / eq:varE2), F_i = p f(i/p), F_{p-1} = -sum_i F_i,
     sigma(E2)/sigma(E1)   (empirical amplification).

 sigma(E1) is function-independent (Prop estimation1-erreur): we therefore
 measure it ONCE (standboot only, KE1 trials) and run lutboot ALONE per function
 for sigma(E2). This avoids a redundant standboot per function -- the main
 speed-up over a naive two-bootstraps-per-trial loop (~2x fewer bootstraps),
 plus fewer random functions and modest per-function K.

 Output: bench_lut_amplification.csv  
 REQUIRES: multival_boot.jl (self-contained).  RUN: julia -t auto bench_lut_amplification.jl
=============================================================================#

include("multival_boot.jl")
using Printf, Statistics, Random, Base.Threads

sigmin(N) = 2.0^(1 - 0.0254*N)

# f : T_p -> T_p by integer values F_0..F_{p-2}; last fixed by cyclicity (sum 0)
function fvals_from(G::Vector{Int}, p::Int)
    Float64.(vcat(G, -sum(G))) ./ p
end
function named_G(name::Symbol, p::Int)
    base = [mods(i, p) for i in 0:p-2]
    name === :identity ? base :
    name === :sign     ? sign.(base) :
    name === :relu     ? max.(base, 0) :
    name === :square   ? [mods(i*i, p) for i in 0:p-2] :
    name === :compare  ? [b > 0 ? 1 : 0 for b in base] :
    error("unknown $name")
end

Vf_trace(c, fvals) = Vf_norm(c, fvals)                       # from lutboot.jl
newfactor(c, fvals) = sqrt(sum(abs2, round.(Int, c.t .* fvals)) / 2)  # sqrt((sum F_i^2)/2)

function run(; p = 11, alpha = 3, Dlog = 8, ell = 3,
             K = 200, Krand = 120, KE1 = 300, nrand = 12, seed = 11)
    N = (p-1)*p^(alpha-1); sbk = sigmin(N)
    c = make(p, p, alpha, 1, 0.0; sig_bk = sbk, Dlog = Dlog, ell = ell)
    setup = Xoshiro(seed); s, sT = keygen(c, setup)
    BK = bootstrap_keys(c, WS(c), s, sT, c.sig_bk, setup)
    half = (p-1)÷2

    @printf("== amplification bench | p=%d N=%d sig_BK=2^%.1f D=2^%d ell=%d ==\n",
            p, N, log2(sbk), Dlog, ell)

    # --- sigma(E1): FUNCTION-INDEPENDENT, so measure ONCE (standboot only) ------
    fid = fvals_from(named_G(:identity, p), p)
    e1 = zeros(KE1)
    @threads for k in 1:KE1
        r = Xoshiro(5_000_000 + k)
        i0 = rand(r, 0:p-1); mu = mods(i0, p)/p; tgt = fid[i0+1]
        a, b = lwe_enc(c, mu, s, 0.0, r)
        as, bs_ = standboot(c, a, b, s, BK, fid)
        e1[k] = mod1c(lwe_phase(as, bs_, s) - tgt)
    end
    sE1 = std(e1)
    @printf("pooled sigma(E1) = %.4e  (%d standboots, function-independent)\n", sE1, KE1)

    # --- sigma(E2) per function: lutboot ONLY ----------------------------------
    jobs = Tuple{String,Vector{Int},Int}[]
    for nm in (:identity, :sign, :relu, :square, :compare)
        push!(jobs, (String(nm), named_G(nm, p), K))
    end
    for rr in 1:nrand
        push!(jobs, ("rand$rr", rand(Xoshiro(1000 + rr), -half:half, p-1), Krand))
    end

    @printf("%-9s %10s %10s %12s %12s\n", "func", "||V||_Tr", "newfactor", "s(E2)", "s(E2)/s(E1)")
    rows = String["func,Vtr,newfactor,sE2,ratio"]
    for (jx, (name, G, Kf)) in enumerate(jobs)
        fvals = fvals_from(G, p)
        Vtr = Vf_trace(c, fvals); nf = newfactor(c, fvals)
        RG, c0 = lut_key(c, sT, fvals, setup)
        e2 = zeros(Kf)
        @threads for k in 1:Kf
            r = Xoshiro(7_000_000 + 1000*jx + k)
            i0 = rand(r, 0:p-1); mu = mods(i0, p)/p; tgt = fvals[i0+1]
            a, b = lwe_enc(c, mu, s, 0.0, r)
            al, bl = lutboot(c, a, b, s, BK, RG, c0)
            e2[k] = mod1c(lwe_phase(al, bl, s) - tgt)
        end
        sE2 = std(e2); ratio = sE2 / sE1
        startswith(name, "rand") || @printf("%-9s %10.3f %10.3f %12.3e %12.3f\n",
                                            name, Vtr, nf, sE2, ratio)
        push!(rows, @sprintf("%s,%.6f,%.6f,%.6e,%.6f", name, Vtr, nf, sE2, ratio))
    end
    open("bench_lut_amplification.csv", "w") do io
        for r in rows; println(io, r); end
    end
    println("wrote bench_lut_amplification.csv  (pooled sE1 = ", @sprintf("%.4e", sE1), ")")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
