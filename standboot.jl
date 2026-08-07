#=============================================================================
 standboot.jl  --  SELF-CONTAINED single-value (SVM) and multi-value (MVM)
 bootstrap over the prime-power cyclotomic ring M = t^alpha, with the final LWE
 ciphertext obtained through the CLOSED-FORM extraction formulas of
 Proposition 7 (and its dual Remark 6) of [CKL25], derived once and for all,
 instead of forming the trace explicitly.

 No include, no external package: pure Base Julia + stdlib (Random, Statistics,
 Printf, Base.Threads), hand-rolled radix-2 FFT.

 Extraction (case r = 1, M = t^alpha, m = M/t, indices mod M):

   (SVM, dual accumulator)   LWE of Tr(acc*) = dual coordinate 0:
       b^   = b*_0
       a^_j = a*_{-j} - [1 <= j <= m] P_{m-j},    P_s = sum_{r = s mod m} a*_r

   (MVM, primal accumulator) LWE of coordinate 0 of V_f*.acc:
       b^   = sum_i F_i b_{-i m}     - S b_m
       a^_j = sum_i F_i a_{-i m - j} - S a_{m-j},   S = sum_i F_i, F_i = p f(i/p)

 The MVM formula is the key gain: since Tr(X^{-i} v_f*) = <Vbar_f*, X^{-i} w> is
 a LINEAR functional of the accumulator, the per-function factor V_f* need not be
 applied homomorphically at all -- it is folded into the extraction. This removes
 the ell gadget products of extprod_star AND the RGSW*(V_f*) key itself, at a cost
 of O(N p) scalar operations. It is also exactly the model of the paper's error
 estimate, in which E_2 = <Vbar_f*, Err(c)>.

 Contents: FFT + ring arithmetic -> LWE/RLWE/RGSW -> blind rotation (dual and
 non-dual) -> reference trace extraction and RGSW* lift (kept as the baseline of
 the self-tests) -> Proposition-7 extraction -> SVM/MVM bootstraps -> tests.

 RUN:  julia standboot.jl          # selftest_bis() + bench_bis()
=============================================================================#

using Random, Statistics, Printf, Base.Threads

mod1c(x::Float64) = x - round(x)
mods(x::Integer, q::Integer) = (r = mod(x, q); r > (q - 1) ÷ 2 ? r - q : r)

# ------------------------------- FFT machinery -------------------------------
function bitrev_perm(L::Int)
    bits = trailing_zeros(L)
    br = Vector{Int}(undef, L)
    for i in 0:L-1
        r = 0; x = i
        for _ in 1:bits
            r = (r << 1) | (x & 1); x >>= 1
        end
        br[i+1] = r + 1
    end
    br
end

"in-place iterative radix-2 FFT; ws[j+1] = exp(-2*pi*i*j/L), j = 0..L/2-1"
function fft!(x::Vector{ComplexF64}, ws::Vector{ComplexF64}, br::Vector{Int}, inverse::Bool)
    L = length(x)
    @inbounds for i in 1:L
        j = br[i]
        if j > i
            x[i], x[j] = x[j], x[i]
        end
    end
    len = 2
    @inbounds while len <= L
        half = len >> 1
        step = L ÷ len
        for base in 0:len:L-1
            for k in 0:half-1
                w = ws[k*step+1]
                inverse && (w = conj(w))
                u = x[base+k+1]
                v = x[base+half+k+1] * w
                x[base+k+1] = u + v
                x[base+half+k+1] = u - v
            end
        end
        len <<= 1
    end
    if inverse
        @inbounds for i in 1:L
            x[i] /= L
        end
    end
    x
end

# ------------------------------- configuration -------------------------------
struct Cfg
    p::Int; t::Int; alpha::Int; mc::Int
    m::Int; M::Int; N::Int; n::Int; nb::Int; Mt::Int
    bs::Vector{Int}; boff::Vector{Int}     # block sizes / offsets (ragged collapse)
    D::Int; ell::Int
    sig_e::Float64; sig_bk::Float64
    fvec::Vector{Float64}
    L::Int; ws::Vector{ComplexF64}; br::Vector{Int}
end

function make(p, t, alpha, mc, sig_e; sig_bk = 2.0^-24, Dlog = 7, ell = 3)
    m = t^(alpha-1); M = t^alpha; N = (t-1)*m
    bs = vcat(fill(mc, N ÷ mc), fill(1, N % mc))   # trailing components ungrouped
    boff = vcat(0, cumsum(bs)[1:end-1])
    Mt = p*t
    fvec = zeros(M)
    for i in 0:t-2, j in 0:m-1
        fvec[i*m + j + 1] = mods(i, t)/Mt          # v*_{im+j} = f(i/t)   (Prop. 5)
    end
    L = 1
    while L < 2M - 1; L <<= 1; end
    ws = [cis(-2pi*j/L) for j in 0:L÷2-1]
    Cfg(p, t, alpha, mc, m, M, N, N, length(bs), Mt, bs, boff,
        2^Dlog, ell, sig_e, sig_bk, fvec, L, ws, bitrev_perm(L))
end

mutable struct WS                                   # per-trial FFT workspace
    A::Vector{ComplexF64}; B::Vector{ComplexF64}
    tmp::Vector{Float64}; nA::Vector{Float64}; nB::Vector{Float64}
end
WS(c::Cfg) = WS(zeros(ComplexF64, c.L), zeros(ComplexF64, c.L),
                zeros(c.M), zeros(c.M), zeros(c.M))

# --------------------------- ring operations ---------------------------------
"cyclic convolution of length M via pad-and-fold FFT, result into `out`"
function cconv!(c::Cfg, W::WS, out::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64})
    fill!(W.A, 0.0im); fill!(W.B, 0.0im)
    @inbounds for i in 1:c.M
        W.A[i] = u[i]; W.B[i] = v[i]
    end
    fft!(W.A, c.ws, c.br, false); fft!(W.B, c.ws, c.br, false)
    @inbounds for i in 1:c.L
        W.A[i] *= W.B[i]
    end
    fft!(W.A, c.ws, c.br, true)
    @inbounds for q in 0:c.M-1                      # fold the linear conv mod M
        s = real(W.A[q+1])
        q <= c.M - 2 && (s += real(W.A[q + c.M + 1]))
        out[q+1] = s
    end
    out
end

function cconv_naive(c::Cfg, u, v)                  # for test T0c only
    out = zeros(c.M)
    for s in 0:c.M-1
        v[s+1] == 0.0 && continue
        for r in 0:c.M-1
            out[mod(r + s, c.M) + 1] += u[r+1]*v[s+1]
        end
    end
    out
end

"partial sums P_s = sum_{r == s mod m} u_r"
function psums(c::Cfg, u::Vector{Float64})
    P = zeros(c.m)
    @inbounds for r in 0:c.M-1
        P[mod(r, c.m) + 1] += u[r+1]
    end
    P
end

"dual poly d  x  T-side poly cc  -> dual poly, into `out`  (general alpha)"
function dmul!(c::Cfg, W::WS, out::Vector{Float64}, d::Vector{Float64}, cc::Vector{Float64})
    cconv!(c, W, out, d, cc)
    P = psums(c, d)
    @inbounds for sp in 0:c.m-1
        coef = P[sp+1]
        coef == 0.0 && continue
        for q in 0:c.M-1
            out[q+1] -= coef * cc[mod(q + c.m - sp, c.M) + 1]
        end
    end
    @inbounds for q in c.N:c.M-1                    # discard the upper block
        out[q+1] = 0.0
    end
    out
end

"dual: multiply by the monomial X^k (tests and extraction derivation)"
function mxpow(c::Cfg, u::Vector{Float64}, k::Int)
    out = zeros(c.M)
    @inbounds for q in 0:c.M-1
        out[mod(q + k, c.M) + 1] = u[q+1]
    end
    P = psums(c, u)
    for sp in 0:c.m-1
        out[mod(k - c.m + sp, c.M) + 1] -= P[sp+1]
    end
    for q in c.N:c.M-1
        out[q+1] = 0.0
    end
    out
end

rotc(c::Cfg, cc::Vector{Float64}, k::Int) =
    [cc[mod(i - k, c.M) + 1] for i in 0:c.M-1]      # T-side X^k (cyclic shift)

# --------------------------- LWE / RLWE / RGSW -------------------------------
function keygen(c::Cfg, r::AbstractRNG)
    s = rand(r, 0:1, c.n)
    sT = zeros(c.M); sT[1:c.N] .= s
    s, sT
end

function lwe_enc(c::Cfg, mu::Float64, s::Vector{Int}, sig::Float64, r::AbstractRNG)
    a = rand(r, c.n) .- 0.5
    ph = sum(a[i]*s[i] for i in 1:c.n) + mu + sig*randn(r)
    a, mod1c(ph)
end
lwe_phase(a, b, s) = mod1c(b - sum(a[i]*s[i] for i in eachindex(s)))
decrypt(c::Cfg, a, b, s) = mod(round(Int, c.p*lwe_phase(a, b, s)), c.p)

function rlwe_enc(c::Cfg, W::WS, mu::Vector{Float64}, sT::Vector{Float64},
                  sig::Float64, r::AbstractRNG)
    a = rand(r, c.M) .- 0.5
    sa = zeros(c.M); cconv!(c, W, sa, sT, a)
    b = mod1c.(sa .+ mu .+ sig .* randn(r, c.M))
    a, b
end

const Rows = Vector{Tuple{Vector{Float64},Vector{Float64}}}

"RGSW of the integer constant msg: rows RLWE(-s msg/D^t) then RLWE(msg/D^t)"
function rgsw_enc(c::Cfg, W::WS, msg::Int, sT::Vector{Float64}, sig::Float64, r::AbstractRNG)
    r1 = Rows(); r2 = Rows()
    for t in 1:c.ell
        Dt = float(c.D)^t
        push!(r1, rlwe_enc(c, W, mod1c.(-msg .* sT ./ Dt), sT, sig, r))
        c0 = zeros(c.M); c0[1] = msg/Dt
        push!(r2, rlwe_enc(c, W, c0, sT, sig, r))
    end
    r1, r2
end

newrot(c::Cfg, C, k::Int) = ([(rotc(c,a,k), rotc(c,b,k)) for (a,b) in C[1]],
                             [(rotc(c,a,k), rotc(c,b,k)) for (a,b) in C[2]])

"H += X^k . C  (in place on H)"
function addrot!(c::Cfg, H, C, k::Int)
    for blk in 1:2, t in 1:c.ell, comp in 1:2
        h = H[blk][t][comp]; g = C[blk][t][comp]
        @inbounds for i in 0:c.M-1
            h[i+1] = mod1c(h[i+1] + g[mod(i - k, c.M) + 1])
        end
    end
    H
end

"balanced base-D digits of the dual coordinates"
function decomp(c::Cfg, u::Vector{Float64})
    digs = [zeros(c.M) for _ in 1:c.ell]
    w = round.(Int64, u .* float(c.D)^c.ell)
    for t in c.ell:-1:1
        @inbounds for q in 1:c.M
            r = mod(w[q], c.D)
            r > c.D ÷ 2 && (r -= c.D)
            digs[t][q] = r
            w[q] = (w[q] - r) ÷ c.D
        end
    end
    digs
end

"external product  ACC <- RGSW(H) box ACC   (dual version, Subsection 3.4)"
function ext_prod!(c::Cfg, W::WS, H, accA::Vector{Float64}, accB::Vector{Float64})
    dA, dB = decomp(c, accA), decomp(c, accB)
    fill!(W.nA, 0.0); fill!(W.nB, 0.0)
    for t in 1:c.ell
        dmul!(c, W, W.tmp, dA[t], H[1][t][1]); W.nA .+= W.tmp
        dmul!(c, W, W.tmp, dB[t], H[2][t][1]); W.nA .+= W.tmp
        dmul!(c, W, W.tmp, dA[t], H[1][t][2]); W.nB .+= W.tmp
        dmul!(c, W, W.tmp, dB[t], H[2][t][2]); W.nB .+= W.tmp
    end
    accA .= mod1c.(W.nA)
    accB .= mod1c.(W.nB)
end

# --------------------- Section 7 procedure, with collapse --------------------
block_pat(c::Cfg, s, k) = sum(s[c.boff[k] + i] << (i-1) for i in 1:c.bs[k])

function bootstrap_keys(c::Cfg, W::WS, s, sT, sig, r::AbstractRNG)
    [[rgsw_enc(c, W, pat == block_pat(c, s, k) ? 1 : 0, sT, sig, r)
      for pat in 0:2^c.bs[k]-1] for k in 1:c.nb]
end

"abar_{k,jtilde} = round( Mt * atilde_k . jtilde - delta_{k,1} Mt b )   (eq. (18)-(19))"
function abar(c::Cfg, a, b, k::Int, pat::Int)
    dot = 0.0
    for i in 1:c.bs[k]
        ((pat >> (i-1)) & 1) == 1 && (dot += a[c.boff[k] + i])
    end
    round(Int, c.Mt*dot - (k == 1 ? c.Mt*b : 0.0))
end

"the whole refreshing procedure"
function refresh(c::Cfg, a, b::Float64, i_msg::Int, s, sT, BK)
    W = WS(c)
    ab = [[abar(c, a, b, k, pat) for pat in 0:2^c.bs[k]-1] for k in 1:c.nb]
    S = sum(ab[k][block_pat(c, s, k) + 1] for k in 1:c.nb)
    E = mods(-S - i_msg*c.t, c.Mt)
    accA, accB = zeros(c.M), copy(c.fvec)
    for k in 1:c.nb                                   # exponents rescaled by m = M/t
        H = newrot(c, BK[k][1], mod(c.m*ab[k][1], c.M))
        for pat in 1:2^c.bs[k]-1
            addrot!(c, H, BK[k][pat+1], mod(c.m*ab[k][pat+1], c.M))
        end
        ext_prod!(c, W, H, accA, accB)
    end
    bh = accB[1]                                      # Tr = dual coordinate 0
    P = psums(c, accA)
    ah = zeros(c.n)
    for i in 0:c.N-1
        ah[i+1] = accA[mod(c.M - i, c.M) + 1] - (1 <= i <= c.m ? P[c.m - i + 1] : 0.0)
    end
    a2 = mod1c.(round.(c.Mt .* a) ./ c.Mt .- ah)
    b2 = mod1c(round(c.Mt*b)/c.Mt - bh)
    E, (ah, bh), (a2, b2)
end

# ------------------------------- predictions ---------------------------------
function erfc_(x::Float64)
    x < 0 && return 2.0 - erfc_(-x)
    t = 1.0/(1.0 + 0.3275911x)
    t*(0.254829592 + t*(-0.284496736 + t*(1.421413741 +
        t*(-1.453152027 + t*1.061405429))))*exp(-x*x)
end

function pfail_31(lam, Mt, L, sig)
    sigchi = sqrt(sig^2 + L/(12.0*Mt^2))
    T = 42.0/sigchi
    ns = max(ceil(Int, T/(pi/(60lam))), 20000)
    isodd(ns) && (ns += 1)
    h = T/ns
    f(tt) = tt == 0.0 ? lam :
        (x = tt/(2Mt); sin(lam*tt)/tt*(sin(x)/x)^L*exp(-sig^2*tt^2/2))
    ssum = f(0.0) + f(T)
    for j in 1:ns-1
        ssum += (isodd(j) ? 4 : 2)*f(j*h)
    end
    1.0 - (2/pi)*(h/3)*ssum
end

pfail_clt(lam, Mt, L, sig) = erfc_(lam/(sqrt(2)*sqrt(sig^2 + L/(12.0*Mt^2))))

sigout_pred(c::Cfg) = sqrt(sum((c.ell*c.N*c.D^2*2^b/3.0)*c.sig_bk^2 for b in c.bs) +
                           c.nb*(1 + c.n/2)*float(c.D)^(-2c.ell)/12.0)   # 1 + n/2 (key Hamming)

# --------------------------------- test battery ------------------------------
function run(c::Cfg, K::Int, label::String)
    setup = Xoshiro(20260709)
    s, sT = keygen(c, setup)
    L = count(k -> block_pat(c, s, k) != 0, 2:c.nb)
    @printf("== %s: p=%d t=%d alpha=%d mc=%d M=%d N=n=%d nb=%d (%dx%d + %dx1) Mt=%d sig_e=%.3f L=%d  [%d threads]\n",
            label, c.p, c.t, c.alpha, c.mc, c.M, c.N, c.nb,
            c.N ÷ c.mc, c.mc, c.N % c.mc, c.Mt, c.sig_e, L, nthreads())
    W = WS(c)

    u = randn(setup, c.M); v = randn(setup, c.M)
    out = zeros(c.M); cconv!(c, W, out, u, v)
    efft = maximum(abs.(out .- cconv_naive(c, u, v)))
    @printf("  [T0c] FFT cconv vs naive: max err = %.2e -> %s\n", efft, efft < 1e-9 ? "PASS" : "FAIL")

    Emax = (c.t - 1) ÷ 2
    err = maximum(abs(mxpow(c, c.fvec, -c.m*E0)[1] - mods(E0, c.t)/c.Mt)
                  for E0 in -Emax:Emax)
    @printf("  [T0]  Theta table: max err = %.2e -> %s\n", err, err < 1e-12 ? "PASS" : "FAIL")

    BK0 = bootstrap_keys(c, W, s, sT, 0.0, setup)
    ok, mxc = true, 0.0
    for _ in 1:40
        i_msg = rand(setup, 0:c.p-1)
        a, b = lwe_enc(c, mods(i_msg, c.p)/c.p, s, 0.15/(2c.p), setup)
        E, (ah, bh), (a2, b2) = refresh(c, a, b, i_msg, s, sT, BK0)
        mxc = max(mxc, abs(mod1c(lwe_phase(ah, bh, s) - mods(E, c.t)/c.Mt)))
        ok &= (abs(E) > Emax) || (decrypt(c, a2, b2, s) == i_msg)
    end
    @printf("  [T1]  noise-free pipeline: max corr err = %.2e, conditional decrypt -> %s\n",
            mxc, (ok && mxc < 1e-4) ? "PASS" : "FAIL")

    BK = bootstrap_keys(c, W, s, sT, c.sig_bk, setup)
    condok = falses(K); ce = fill(NaN, K); chis = zeros(K)
    @threads for trial in 1:K
        r = Xoshiro(0x9E3779B97F4A7C15*UInt64(trial) + 0x00C0FFEE)
        i_msg = rand(r, 0:c.p-1)
        a, b = lwe_enc(c, mods(i_msg, c.p)/c.p, s, c.sig_e, r)
        E, (ah, bh), _ = refresh(c, a, b, i_msg, s, sT, BK)
        cond = abs(E) <= Emax
        condok[trial] = cond
        cond && (ce[trial] = mod1c(lwe_phase(ah, bh, s) - mods(E, c.t)/c.Mt))
        U = 0.0
        for k in 2:c.nb
            bp = block_pat(c, s, k)
            bp == 0 && continue
            dot = sum(s[c.boff[k] + i]*a[c.boff[k] + i] for i in 1:c.bs[k])
            U += abar(c, a, b, k, bp) - c.Mt*dot
        end
        chis[trial] = mod1c(lwe_phase(a, b, s) - mods(i_msg, c.p)/c.p) - U/c.Mt
    end
    fails = K - count(condok)
    lam = 1.0/(2c.p)
    q31, qclt = pfail_31(lam, c.Mt, L, c.sig_e), pfail_clt(lam, c.Mt, L, c.sig_e)
    z = (fails - K*q31)/sqrt(K*q31*(1 - q31))
    @printf("  [T2]  fail: emp=%.4f (31)=%.4f CLT=%.4f z=%+.2f -> %s\n",
            fails/K, q31, qclt, z, abs(z) < 3 ? "PASS" : "FAIL")

    cev = filter(!isnan, ce); sp = sigout_pred(c)
    ratio = std(cev)/sp
    @printf("  [T3]  out noise: emp=%.3e pred=%.3e ratio=%.2f -> %s\n",
            std(cev), sp, ratio, (0.3 < ratio < 3) ? "PASS" : "FAIL")

    vp = c.sig_e^2 + L/(12.0*c.Mt^2)
    zv = (var(chis) - vp)/(vp*sqrt(2.0/K))
    @printf("  [T4]  chi var: emp=%.3e pred=%.3e z=%+.2f -> %s\n",
            var(chis), vp, zv, abs(zv) < 3.5 ? "PASS" : "FAIL")
end

function testT5()
    p5, Mt5, L5, se5 = 5, 35, 36, 0.004
    lam = 1.0/(2p5)
    q31, qclt = pfail_31(lam, Mt5, L5, se5), pfail_clt(lam, Mt5, L5, se5)
    K5, nch = 10_000_000, 200
    counts = zeros(Int, nch)
    @threads for ch in 1:nch
        r = Xoshiro(0x0000ABCD + 7*UInt64(ch))
        cnt = 0
        for _ in 1:K5 ÷ nch
            S = 0.0
            for _ in 1:L5
                S += rand(r) - 0.5
            end
            abs(se5*randn(r) - S/Mt5) > lam && (cnt += 1)
        end
        counts[ch] = cnt
    end
    fails = sum(counts)
    z31 = (fails - K5*q31)/sqrt(K5*q31*(1 - q31))
    zclt = (fails - K5*qclt)/sqrt(K5*qclt*(1 - qclt))
    @printf("== T5 rounding-dominated (Mt=%d, L=%d, sig_e=%.3f): emp=%.5f | (31)=%.5f z=%+.2f | CLT=%.5f z=%+.2f\n",
            Mt5, L5, se5, fails/K5, q31, z31, qclt, zclt)
    println("   -> ", abs(z31) < 3 < abs(zclt) ? "PASS (formula (31) accepted, CLT rejected)" :
            (abs(z31) < 3 ? "PASS" : "FAIL"))
end

function main()
    @time run(make(5, 67, 1, 1, 0.046), 3000, "A (alpha=1, mc=1)")
    @time run(make(5, 67, 1, 2, 0.046), 3000, "C (alpha=1, mc=2)")
    @time run(make(5,  7, 2, 1, 0.010), 4000, "B (alpha=2, M=49, mc=1)")
    @time run(make(5,  7, 2, 3, 0.010), 4000, "D (alpha=2, M=49, mc=3)")
    @time testT5()
end

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

# --------------------------------------------------------------------------
#  Extraction plans: index tables derived once per configuration.
# --------------------------------------------------------------------------

"Tables for the (SVM) dual extraction of Proposition 7 / Remark 6 of [CKL25]."
struct SVMPlan
    idx::Vector{Int}          # idx[j+1] = ((M - j) mod M) + 1     (1-based)
    cor::Vector{Int}          # cor[j+1] = m-j (0-based) or 0 if no correction
end
function SVMPlan(c::Cfg)
    idx = [mod(c.M - j, c.M) + 1 for j in 0:c.N-1]
    cor = [(1 <= j <= c.m) ? (c.m - j + 1) : 0 for j in 0:c.N-1]   # 1-based into P
    SVMPlan(idx, cor)
end

"Tables for the (MVM) generalised extraction <Vbar_f*, .> of Proposition 7."
struct MVMPlan
    F::Vector{Float64}        # F_i = p f(i/p),  i = 0..p-2
    S::Float64                # S = sum_i F_i
    off::Vector{Int}          # off[i+1] = -i*m  (extraction reads coord -i*m)
    c0::Float64               # constant subtracted from f, added back at the end
end
function MVMPlan(c::Cfg, fvals::Vector{Float64})
    c0 = sum(fvals)/length(fvals)
    F  = [c.t*(fvals[i+1] - c0) for i in 0:c.t-2]
    nonint = maximum(abs.(F .- round.(F)))
    nonint > 1e-9 && @warn @sprintf(
        "MVMPlan: V_f* is NON-INTEGRAL (max frac %.3g); MVM is exact only for LUT outputs on the 1/p grid.",
        nonint)
    MVMPlan(F, sum(F), [-i*c.m for i in 0:c.t-2], c0)
end

# --------------------------------------------------------------------------
#  Closed-form extractions
# --------------------------------------------------------------------------

"(SVM) LWE of the trace (dual coordinate 0) of the RLWE* accumulator, via the
 precomputed plan. Identical to trace_extract, but with no per-call index work."
function extract_svm(c::Cfg, pl::SVMPlan, accA::Vector{Float64}, accB::Vector{Float64})
    P = psums(c, accA)
    ah = Vector{Float64}(undef, c.n)
    @inbounds for j in 1:c.N
        v = accA[pl.idx[j]]
        pl.cor[j] != 0 && (v -= P[pl.cor[j]])
        ah[j] = v
    end
    ah, mod1c(accB[1])
end

"(MVM) LWE of <Vbar_f*, acc> directly from the NON-DUAL accumulator: the factor
 V_f* is applied at extraction time, so no external product and no RGSW*(V_f*).
      a^_j = sum_i F_i a_{-i m - j} - S a_{m-j},  b^ = sum_i F_i b_{-i m} - S b_m."
function extract_mvm(c::Cfg, pl::MVMPlan, accA::Vector{Float64}, accB::Vector{Float64})
    M, m = c.M, c.m
    ah = zeros(Float64, c.n)
    @inbounds for j in 0:c.N-1
        acc = 0.0
        for i in eachindex(pl.F)
            Fi = pl.F[i]
            Fi == 0.0 && continue
            acc += Fi * accA[mod(pl.off[i] - j, M) + 1]      # off[i] = -i*m
        end
        ah[j+1] = acc - pl.S * accA[mod(m - j, M) + 1]
    end
    bh = 0.0
    @inbounds for i in eachindex(pl.F)
        bh += pl.F[i] * accB[mod(pl.off[i], M) + 1]
    end
    bh -= pl.S * accB[mod(m, M) + 1]
    ah, mod1c(bh)
end

# --------------------------------------------------------------------------
#  The two bootstraps, Proposition-7 flavour
# --------------------------------------------------------------------------

"(SVM) standard bootstrap: rotate v_f*, extract by the closed form."
function standboot_bis(c::Cfg, a, b::Float64, s, BK, fvals::Vector{Float64},
                       pl::SVMPlan = SVMPlan(c))
    c0 = sum(fvals)/length(fvals)
    accA, accB = blindrot(c, a, b, s, BK, testpoly_f(c, fvals .- c0))
    ah, bh = extract_svm(c, pl, accA, accB)
    ah, mod1c(bh + c0)
end

"(MVM) multi-value bootstrap: ONE function-independent rotation of w, then the
 generalised extraction. No RGSW*(V_f*), no external product."
function lutboot_bis(c::Cfg, a, b::Float64, s, BK, mk::MVMPlan)
    accA, accB = blindrot_nd(c, a, b, s, BK, make_w(c))
    ah, bh = extract_mvm(c, mk, accA, accB)
    ah, mod1c(bh + mk.c0)
end

# --------------------------------------------------------------------------
#  Self-tests: the closed forms must reproduce the reference pipelines
# --------------------------------------------------------------------------

"Compare (a) extract_svm vs trace_extract, and (b) lutboot_bis vs standboot,
 slot by slot, in a noise-free configuration where both must be exact."
function selftest_bis(; p = 5, bits = 80, mc = 2, Dlog = 15, ell = 2, func = :sign)
    c = make(p, p, alpha_for(p, bits), mc, 0.0; sig_bk = 2.0^-52, Dlog = Dlog, ell = ell)
    setup = Xoshiro(11); s, sT = keygen(c, setup)
    BK = bootstrap_keys(c, WS(c), s, sT, c.sig_bk, setup)
    fvals = make_fvals(func, c.t)
    pl = SVMPlan(c); mk = MVMPlan(c, fvals)
    okA = okB = true
    @printf("selftest_bis (p=%d, M=%d, N=%d, f=%s)\n", p, c.M, c.N, func)
    @printf("%4s %10s %12s %12s %12s\n", "i0", "target", "SVM_bis", "MVM_bis", "SVM_ref")
    for i0 in 0:c.t-1
        mu = mods(i0, c.t)/c.t; tgt = fvals[i0+1]
        a, b = lwe_enc(c, mu, s, 0.0, setup)
        a1, b1 = standboot_bis(c, a, b, s, BK, fvals, pl)
        a2, b2 = lutboot_bis(c, a, b, s, BK, mk)
        a3, b3 = standboot(c, a, b, s, BK, fvals)              # reference
        v1 = mod1c(lwe_phase(a1,b1,s)); v2 = mod1c(lwe_phase(a2,b2,s))
        v3 = mod1c(lwe_phase(a3,b3,s))
        abs(mod1c(v1-tgt)) < 1e-3 || (okA = false)
        abs(mod1c(v2-tgt)) < 1e-3 || (okB = false)
        @printf("%4d %+10.4f %+12.4f %+12.4f %+12.4f\n", i0, tgt, v1, v2, v3)
    end
    @printf("SVM (Prop.7 extraction) = %s     MVM (Prop.7 extraction) = %s\n",
            okA ? "OK" : "BUG", okB ? "OK" : "BUG")
    okA && okB
end

"Timing: MVM with the Prop.-7 extraction vs the RGSW*(V_f*) external product."
function bench_bis(; p = 11, bits = 128, mc = 1, Dlog = 8, ell = 3, K = 20)
    c = make(p, p, alpha_for(p, bits), mc, 0.0; sig_bk = 2.0^-30, Dlog = Dlog, ell = ell)
    setup = Xoshiro(7); s, sT = keygen(c, setup)
    BK = bootstrap_keys(c, WS(c), s, sT, c.sig_bk, setup)
    fvals = make_fvals(:sign, c.t)
    mk = MVMPlan(c, fvals)
    # reference key RGSW*(V_f*), built inline (lut_key lives in lutboot.jl, which
    # this file deliberately does not include)
    c0 = sum(fvals)/length(fvals)
    RG = rgsw_star_enc(c, WS(c), make_Vf(c, fvals .- c0), sT, 0.0, setup)
    a, b = lwe_enc(c, 0.0, s, 0.0, setup)
    accA, accB = blindrot_nd(c, a, b, s, BK, make_w(c))        # shared rotation
    extract_mvm(c, mk, accA, accB); extprod_star(c, WS(c), RG, accA, accB)  # warm-up
    t1 = @elapsed for _ in 1:K; extract_mvm(c, mk, accA, accB); end
    t2 = @elapsed for _ in 1:K
        a2, b2 = extprod_star(c, WS(c), RG, accA, accB); trace_extract(c, a2, b2)
    end
    @printf("per-function cost after the shared rotation (p=%d, M=%d, ell=%d, K=%d):\n",
            p, c.M, ell, K)
    @printf("   Prop.7 extraction      : %8.3f ms\n", 1e3*t1/K)
    @printf("   RGSW* product + extract: %8.3f ms   -> speed-up x%.1f\n", 1e3*t2/K, t2/t1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    selftest_bis()
    println()
    bench_bis()
end
