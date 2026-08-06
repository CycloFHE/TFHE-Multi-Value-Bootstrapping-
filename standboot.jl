#=============================================================================
 This program was built with the assistance of the AI Claude.

 standboot.jl -- STANDARD (direct) bootstrap: the function f is baked into the
 test polynomial v_f*, the blind rotation rotates v_f*, and a plain TRACE
 EXTRACTION <1,.> reads off f of the message. One blind rotation per function.

 Output error: E1 (blind-rotation noise). Compare with lutboot.jl, the LUT /
 multi-value mode (rotate a shared w, external product by V_f*, extract), whose
 output error is E2 = ||V_f*||_Tr * E1.

 The constant c0 = mean(f) is subtracted so f - c0 satisfies the cyclicity
 condition sum_k f(k/p) = 0 (which lifts the negacyclicity restriction), and
 added back to the output after the bootstrap.

 This file holds the SHARED primitives (functions, test polynomial, PBS phase,
 blind rotation, trace extraction) and the standard bootstrap; lutboot.jl
 includes it.

 REQUIRES: refresh_bootstrap.jl.
 RUN:      julia standboot.jl
=============================================================================#

include("refresh_bootstrap.jl")
using Printf, Statistics, Random

# ---- security-compatible ring: M = p^alpha, N = (p-1)p^{alpha-1} -------------
# 128-bit (N >~ 2000): p=5->a=5 (N=2500); p=7->a=4 (N=2058); p=11->a=3 (N=1210).
# 80-bit  starter (fast): p=5->a=4 (N=500, M=625); p=7->a=3 (N=294); p=11->a=3.
alpha128(p) = p == 5 ? 5 : p == 7 ? 4 : p == 11 ? 3 : error("use p in {5,7,11}")
alpha80(p)  = p == 5 ? 4 : p == 7 ? 3 : p == 11 ? 3 : error("use p in {5,7,11}")
alpha_for(p, bits) = bits == 128 ? alpha128(p) : alpha80(p)

# ---- admissible functions f : T_t -> T_t -------------------------------------
# Each function is specified by its integer values F_i = t f(i/t) in the balanced
# set {-(t-1)/2,...,(t-1)/2} for i = 0..t-2; the LAST value is fixed by the
# cyclicity condition (\ref{cyclicity}) so that sum_i F_i = 0. This guarantees
# F integral AND sum-zero, hence c0 = mean(fvals) = 0 and V_f* = sum_i F_i
# Omega*_{i m} integral -- the torus-exactness precondition of lut_key. (The old
# version scaled some maps by 1/t or used continuous random values, breaking both
# the grid and the sum-zero condition, which made V_f* non-integral after the
# mean subtraction and the LUT bootstrap incorrect.)
function make_fvals(name::Symbol, t::Int)
    base = [mods(i, t) for i in 0:t-2]                  # identity image, i = 0..t-2
    G = name === :id     ? base                          :
        name === :sign   ? sign.(base)                   :
        name === :relu   ? max.(base, 0)                 :
        name === :square ? [mods(i*i, t) for i in 0:t-2] :
        name === :cmp    ? [b > 0 ? 1 : 0 for b in base] :
        name === :rand   ? rand(MersenneTwister(1), -((t-1)÷2):((t-1)÷2), t-1) :
        error("unknown function $name")
    Float64.(vcat(G, -sum(G))) ./ t                     # cyclicity: F_{t-1} = -sum_i F_i
end

"test polynomial v_f* on dual coordinates: v*_{i m + j} = f(i/t), i=0..t-2"
function testpoly_f(c::Cfg, fvals::Vector{Float64})
    v = zeros(c.M)
    for i in 0:c.t-2, j in 0:c.m-1
        v[i*c.m + j + 1] = fvals[i+1]
    end
    v
end

"per-function factor V_f* (dual coords): V_{i m} = t f(i/t), i = 0..t-2"
function make_Vf(c::Cfg, fvals::Vector{Float64})
    V = zeros(c.M)
    for i in 0:c.t-2; V[i*c.m + 1] = c.t*fvals[i+1]; end
    V
end

"function-independent test polynomial w (T-side): w_j = 1/t, j = 0..m-1"
function make_w(c::Cfg)
    w = zeros(c.M)
    for j in 0:c.m-1; w[j+1] = 1.0/c.t; end
    w
end

# ---- PBS phase, blind rotation, trace extraction ----------------------------
# Phase at the ring resolution M = t^alpha, with the shift eps = 1/(2t) - 1/(2M)
# of Proposition prop:testpoly, so the decode point falls at the block CENTRE
# i m + (m-1)/2 (robust to the collapse residual), not the block boundary.
"phase rounding at the PBS resolution M"
function abar_pbs(c::Cfg, a, b, k::Int, pat::Int)
    dot = 0.0
    for i in 1:c.bs[k]
        ((pat >> (i-1)) & 1) == 1 && (dot += a[c.boff[k] + i])
    end
    eps = 1.0/(2*c.t) - 1.0/(2*c.M)
    round(Int, c.M*dot - (k == 1 ? c.M*b + c.M*eps : 0.0))
end

"blind-rotate the (dual) test polynomial by the phase of (a,b); returns (accA, accB)"
function blindrot(c::Cfg, a, b::Float64, s, BK, testpoly::Vector{Float64})
    W = WS(c)
    ab = [[abar_pbs(c, a, b, k, pat) for pat in 0:2^c.bs[k]-1] for k in 1:c.nb]
    accA, accB = zeros(c.M), copy(testpoly)
    for k in 1:c.nb
        H = newrot(c, BK[k][1], mod(ab[k][1], c.M))
        for pat in 1:2^c.bs[k]-1
            addrot!(c, H, BK[k][pat+1], mod(ab[k][pat+1], c.M))
        end
        ext_prod!(c, W, H, accA, accB)
    end
    accA, accB
end

"LWE-encryption of the trace (dual coordinate 0) of the RLWE* accumulator"
function trace_extract(c::Cfg, accA::Vector{Float64}, accB::Vector{Float64})
    P = psums(c, accA); ah = zeros(c.n)
    for i in 0:c.N-1
        ah[i+1] = accA[mod(c.M - i, c.M) + 1] - (1 <= i <= c.m ? P[c.m - i + 1] : 0.0)
    end
    ah, mod1c(accB[1])
end

# ---- non-dual (T-side) rotation, for the LUT mode of lutboot.jl --------------
# w lives in T = R[X]/Phi_M (NON-dual), so rotating it needs the T-side external
# product, not the dual one. The T x T -> T product reduces mod Phi_M using
# Phi_M = sum_{r=0}^{p-1} X^{rm} = 0, i.e. X^{(p-1)m + j} = -sum_{r<p-1} X^{rm+j}.
"T-side product u.v mod Phi_M (cyclic conv, then fold the upper block)"
function tmul!(c::Cfg, W::WS, out::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64})
    cconv!(c, W, out, u, v)
    @inbounds for q in c.N:c.M-1                     # q = (t-1)m + jp,  jp in 0..m-1
        jp = q - (c.t-1)*c.m; val = out[q+1]
        for r in 0:c.t-2; out[r*c.m + jp + 1] -= val; end
        out[q+1] = 0.0
    end
    out
end

"non-dual external product: (T-side acc) box RGSW(H) -> (T-side acc).
 The ring product runs mod X^M-1 (cconv, consistent with the cconv-based keys),
 but we MUST fold mod Phi_M *before* the torus reduction mod1c: the genuine
 T-side element lives in the N-dimensional Phi_M basis {Omega_0..Omega_{N-1}},
 and mod1c (coeff-wise x - round(x)) is the torus reduction only in that basis.
 mod1c and fold_phi! do NOT commute (folding mixes coordinate q in {N..M-1} into
 the {r m + jp} coordinates), and the *next* iteration's decomp is nonlinear, so
 it must see a value already folded and reduced in the correct basis. Deferring
 the fold (the old design) reduced mod 1 in the redundant X^M-1 representation
 and decomposed an unfolded value -- the lutboot correctness bug."
function ext_prod_nd!(c::Cfg, W::WS, H, accA::Vector{Float64}, accB::Vector{Float64})
    dA, dB = decomp(c, accA), decomp(c, accB)
    fill!(W.nA, 0.0); fill!(W.nB, 0.0)
    for t in 1:c.ell
        cconv!(c, W, W.tmp, dA[t], H[1][t][1]); W.nA .+= W.tmp
        cconv!(c, W, W.tmp, dB[t], H[2][t][1]); W.nA .+= W.tmp
        cconv!(c, W, W.tmp, dA[t], H[1][t][2]); W.nB .+= W.tmp
        cconv!(c, W, W.tmp, dB[t], H[2][t][2]); W.nB .+= W.tmp
    end
    # Fold BOTH components mod Phi_M before mod1c so the torus reduction acts in the
    # genuine N-dim T-side basis. NOTE: a residual boundary bug remains in the lift
    # (extprod_star), tracked to non-associativity of dmul! across the Phi_M fold:
    #   dmul(dmul(V,s), a) != dmul(V, tmul(a,s))  at support-crossing slots.
    # With both folded, the lift error is fully characterized as lift = 2*tmul - cconv
    # (extra fold on the a.s term). Fixing dmul! (not the fold choice here) is the
    # correct resolution -- see selftest_basis / the dmul! associativity probe.
    fold_phi!(c, W.nA); fold_phi!(c, W.nB)
    accA .= mod1c.(W.nA); accB .= mod1c.(W.nB)
end

"reduce a length-M coefficient vector mod Phi_M in place (fold the block N..M-1)"
function fold_phi!(c::Cfg, u::Vector{Float64})
    @inbounds for q in c.N:c.M-1
        jp = q - (c.t-1)*c.m; val = u[q+1]
        for r in 0:c.t-2; u[r*c.m + jp + 1] -= val; end
        u[q+1] = 0.0
    end
    u
end

"blind-rotate a T-side (non-dual) test polynomial w by the phase of (a,b).
 Each ext_prod_nd! step folds mod Phi_M and reduces mod 1 in the T-side basis,
 so the accumulator is a proper T-side element (phantom block N..M-1 already
 zero) at every step and the subsequent dual product dmul!(V_f*,.) is exact."
function blindrot_nd(c::Cfg, a, b::Float64, s, BK, testpoly::Vector{Float64})
    W = WS(c)
    ab = [[abar_pbs(c, a, b, k, pat) for pat in 0:2^c.bs[k]-1] for k in 1:c.nb]
    accA, accB = zeros(c.M), copy(testpoly)
    for k in 1:c.nb
        H = newrot(c, BK[k][1], mod(ab[k][1], c.M))
        for pat in 1:2^c.bs[k]-1
            addrot!(c, H, BK[k][pat+1], mod(ab[k][pat+1], c.M))
        end
        ext_prod_nd!(c, W, H, accA, accB)
    end
    accA, accB                                       # folded mod Phi_M, reduced mod 1 (T-side)
end

# ---- proper gadget C*⊠c : lift a non-dual RLWE(mu) to a dual RLWE*(V·mu) -----
# The plaintext dmul! lift mishandles the mask a; the correct C*⊠c decomposes c
# and multiplies by a DUAL RGSW*(V), which carries s correctly. V is public, so
# RGSW*(V) is built noiselessly (sig=0) but still uses the key s in its -V·s rows.
"dual RLWE*-encryption of a dual poly mustar: (a*, b*) with b* = a*·s + mustar + e*"
function rlwe_star_enc(c::Cfg, W::WS, mustar::Vector{Float64}, sT, sig::Float64, r::AbstractRNG)
    a = rand(r, c.M) .- 0.5
    sa = zeros(c.M); dmul!(c, W, sa, a, sT)          # a*·s  (dual product)
    a, mod1c.(sa .+ mustar .+ sig .* randn(r, c.M))
end

"dual RGSW*(Vpoly): gadget rows RLWE*(-Vpoly·s/D^t) and RLWE*(Vpoly/D^t)"
function rgsw_star_enc(c::Cfg, W::WS, Vpoly::Vector{Float64}, sT, sig::Float64, r::AbstractRNG)
    Vs = zeros(c.M); dmul!(c, W, Vs, Vpoly, sT)      # Vpoly·s  (dual)
    r1 = Vector{Tuple{Vector{Float64},Vector{Float64}}}(); r2 = similar(r1)
    for t in 1:c.ell
        Dt = float(c.D)^t
        push!(r1, rlwe_star_enc(c, W, mod1c.(-Vs ./ Dt), sT, sig, r))
        push!(r2, rlwe_star_enc(c, W, mod1c.(Vpoly ./ Dt), sT, sig, r))
    end
    r1, r2
end

"C*⊠c : dec(non-dual RLWE (aA,aB)) · RGSW*(V) -> dual RLWE* (a2,b2) of V·mu"
function extprod_star(c::Cfg, W::WS, RGSWstar, aA::Vector{Float64}, aB::Vector{Float64})
    r1, r2 = RGSWstar; dA, dB = decomp(c, aA), decomp(c, aB)
    na, nb, tmp = zeros(c.M), zeros(c.M), zeros(c.M)
    for t in 1:c.ell
        dmul!(c, W, tmp, r1[t][1], dA[t]); na .+= tmp
        dmul!(c, W, tmp, r2[t][1], dB[t]); na .+= tmp
        dmul!(c, W, tmp, r1[t][2], dA[t]); nb .+= tmp
        dmul!(c, W, tmp, r2[t][2], dB[t]); nb .+= tmp
    end
    mod1c.(na), mod1c.(nb)
end

"standard bootstrap: rotate v_f*, trace extract -> LWE(f(mu)); subtract/add c0"
function standboot(c::Cfg, a, b::Float64, s, BK, fvals::Vector{Float64})
    c0 = sum(fvals)/length(fvals)
    accA, accB = blindrot(c, a, b, s, BK, testpoly_f(c, fvals .- c0))
    ah, bh = trace_extract(c, accA, accB)
    ah, mod1c(bh + c0)
end

# --------------------------------- self-test ----------------------------------
function run_standboot(; p = 5, bits = 80, mc = 2, Dlog = 15, ell = 2, sig_e = 0.0, K = 40)
    c = make(p, p, alpha_for(p, bits), mc, sig_e; sig_bk = 2.0^-40, Dlog = Dlog, ell = ell)
    setup = Xoshiro(11); s, sT = keygen(c, setup)
    BK = bootstrap_keys(c, WS(c), s, sT, c.sig_bk, setup)
    @printf("== standboot: rotate v_f*, trace extract  (M=%d^%d=%d, N=%d, ~%d-bit) ==\n",
            p, alpha_for(p, bits), c.M, c.N, bits)
    @printf("%-8s %6s %14s\n", "f", "ok", "sigma_out (E1)")
    for name in (:id, :sign, :relu, :cmp, :rand)
        fvals = make_fvals(name, c.t); eo = Float64[]; ok = 0
        for _ in 1:K
            i0 = rand(setup, 0:c.t-2); mu = mods(i0, c.t)/c.t; tgt = fvals[i0+1]
            a, b = lwe_enc(c, mu, s, c.sig_e, setup)
            ah, bh = standboot(c, a, b, s, BK, fvals)
            d = mod1c(lwe_phase(ah, bh, s) - tgt); push!(eo, d); abs(d) < 1/(2c.t) && (ok += 1)
        end
        @printf("%-8s %5d%% %14.3e\n", name, round(Int,100ok/K), std(eo))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_standboot()
end
