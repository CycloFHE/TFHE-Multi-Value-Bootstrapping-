#!/usr/bin/env julia
#
# mvm_modes.jl -- the three bootstrapping modes of the paper, in one file.
#
#   SVM      the single-value mode: rotate v_f = (Omega*_0)^{-1} v*_f and read
#            the accumulator through U* = Omega*_0;
#   MVM1     the multi-value mode with the canonical cofactor gamma = 1;
#   MVMpiv   the multi-value mode with the pivot cofactor gamma = M Omega*_0.
#
#
# Standard library only.  Two optional packages are used when present: FFTW, which is what makes ring degrees in the
# thousands reachable, and Plots, without which the numbers are written as CSV.
#
# Usage -- the -t flag sets the number of threads, and the Monte-Carlo loops
# scale with it almost linearly.
#   julia -t auto mvm_modes.jl            run the self-tests
#   julia -t auto mvm_modes.jl run        run the experiments, write the table
#                                         and the three figures
#   julia -t auto mvm_modes.jl run quick  the same, at a reduced sample size
#
# Conventions, fixed once and for all.
#
#   Bases        Omega_i = X^i, i < N, is the monomial basis of R = Z[X]/Phi_M,
#                M = p^alpha, m = M/p, N = (p-1)m.  Its dual for the pairing
#                <P,Q> = Tr(P conj(Q)) is Omega*_r = (X^r - X^{N+[r]_m})/M.
#
#   Noise        every RLWE row is drawn as an RLWE* row -- error spherical in
#                the canonical embedding -- and brought back by (Omega*_0)^{-1}.
#                The RGSW bootstrapping keys are built line by line from those
#                rows, so the key noise law is the same for the three modes.
#
#   Extraction   a functional Tr(U* . ) with U* in R^vee is applied to the
#                primal accumulator, giving an LWE ciphertext under the
#                coefficients of the ring key:
#                    SVM: U* = Omega*_0        MVM: U* = B*_f = gamma V*_f.

using LinearAlgebra, Random, Statistics, Printf

# The torus is carried in Float64 (mod1c keeps representatives in [-1/2,1/2)),
# so ring products are ordinary complex convolutions and FFTW applies directly.
# M = p^alpha has only the small prime factor p, for which FFTW has codelets, so
# the length-M transform is O(M log M) with no zero-padding.  Without FFTW the
# same transform is carried as a dense M x M matrix: exact, dependency-free, and
# perfectly adequate up to M of a few hundred, but O(M^2) beyond.
const HAVE_FFTW = try
    @eval using FFTW
    true
catch
    false
end

# ============================================================== utilities ====

"representative of x in [-1/2, 1/2)"
mod1c(x) = x .- round.(x)

"symmetric representative of k modulo q"
function mods(k::Integer, q::Integer)
    r = mod(k, q)
    return r > (q - 1) ÷ 2 ? r - q : r
end

# =============================================================== the ring ====

"""
    Ring(p, alpha)

R = Z[X]/Phi_M with M = p^alpha, m = M/p, N = (p-1)m, together with the two
bases and the change-of-basis matrices used throughout.

  `V[d,i]`   = tau_d(Omega_i)          the monomial basis in the embeddings
  `Vs[d,r]`  = tau_d(Omega*_r)         the dual basis in the embeddings
  `o0[d]`    = tau_d(Omega*_0)
  `toD`      = Gr(Omega) = real(V' V)  monomial coordinates -> dual coordinates
  `starD`    monomial coordinates of u -> dual coordinates of Omega*_0 u

M = p^alpha is odd and not a power of two.  With FFTW loaded the length-M
transform is used directly -- M has only the small prime factor p, for which FFTW
has codelets -- and `W`, `Wi` stay empty; without it they hold the dense DFT
matrix, which is exact and dependency-free but O(M^2) and confines the file to
M of a few hundred.
"""
struct Ring
    p::Int; alpha::Int; M::Int; m::Int; N::Int
    ds::Vector{Int}                       # the units modulo M
    V::Matrix{ComplexF64}                 # N x N
    Vs::Matrix{ComplexF64}                # N x N
    o0::Vector{ComplexF64}                # N
    toD::Matrix{Float64}                  # N x N
    starD::Matrix{Float64}                # N x N
    W::Matrix{ComplexF64}                 # M x M, forward DFT
    Wi::Matrix{ComplexF64}                # M x M, inverse DFT
end

function Ring(p::Int, alpha::Int)
    M = p^alpha; m = p^(alpha - 1); N = (p - 1) * m
    ds = [d for d in 1:(M-1) if gcd(d, M) == 1]
    @assert length(ds) == N
    V  = ComplexF64[cis(2pi * d * i / M) for d in ds, i in 0:(N-1)]
    Vs = ComplexF64[(cis(2pi * d * r / M) - cis(2pi * d * (N + mod(r, m)) / M)) / M
                    for d in ds, r in 0:(N-1)]
    o0 = ComplexF64[(1 - cis(2pi * d * N / M)) / M for d in ds]
    toD = real(V' * V)
    starD = real(Vs \ (o0 .* V))
    W  = HAVE_FFTW ? zeros(ComplexF64, 0, 0) :
         ComplexF64[cis(-2pi * d * j / M) for d in 0:(M-1), j in 0:(M-1)]
    Wi = HAVE_FFTW ? zeros(ComplexF64, 0, 0) : conj(W) ./ M
    return Ring(p, alpha, M, m, N, ds, V, Vs, o0, toD, starD, W, Wi)
end

"reduce a length-M coefficient vector (or a matrix of them, columns) mod Phi_M"
function fold(R::Ring, u::AbstractVector{<:Real})
    out = Vector{Float64}(undef, R.N)
    @inbounds for i in 0:(R.p - 2), j in 0:(R.m - 1)
        out[i * R.m + j + 1] = u[i * R.m + j + 1] - u[R.N + j + 1]
    end
    return out
end

function fold(R::Ring, U::AbstractMatrix{<:Real})       # columns are elements
    out = Matrix{Float64}(undef, R.N, size(U, 2))
    @inbounds for c in axes(U, 2), i in 0:(R.p - 2), j in 0:(R.m - 1)
        out[i * R.m + j + 1, c] = U[i * R.m + j + 1, c] - U[R.N + j + 1, c]
    end
    return out
end

"zero-pad length-N monomial coordinates to length M"
function pad(R::Ring, u::AbstractVector{<:Real})
    z = zeros(Float64, R.M); z[1:R.N] .= u; return z
end
function pad(R::Ring, U::AbstractMatrix{<:Real})
    Z = zeros(Float64, R.M, size(U, 2)); Z[1:R.N, :] .= U; return Z
end

dft(R::Ring, u)  = HAVE_FFTW ? fft(pad(R, u), 1)  : R.W  * pad(R, u)
idft(R::Ring, U) = fold(R, real(HAVE_FFTW ? ifft(U, 1) : R.Wi * U))

# --- half spectra -----------------------------------------------------------
nfreq(R::Ring) = R.M ÷ 2 + 1

rdft(R::Ring, u::AbstractVector{<:Real}) =
    HAVE_FFTW ? rfft(pad(R, u)) : (R.W * pad(R, u))[1:nfreq(R)]
rdft(R::Ring, U::AbstractMatrix{<:Real}) =
    HAVE_FFTW ? rfft(pad(R, U), 1) : (R.W * pad(R, U))[1:nfreq(R), :]

"the full length-M spectrum of a real signal, rebuilt from its M/2+1 kept bins"
function unhalf(R::Ring, U::AbstractMatrix{<:Complex})
    h = nfreq(R)
    F = Matrix{ComplexF64}(undef, R.M, size(U, 2))
    F[1:h, :] .= U
    @inbounds for c in axes(U, 2), j in (h + 1):R.M
        F[j, c] = conj(U[R.M - j + 2, c])
    end
    return F
end

"the spectrum of X^k on the M/2+1 kept frequencies, as a pointwise multiplier"
shiftspec(R::Ring, k::Integer) =
    ComplexF64[cis(-2pi * mod(k, R.M) * d / R.M) for d in 0:(R.M ÷ 2)]

"u times X^k, on the monomial basis"
xpow(R::Ring, u::AbstractVector{<:Real}, k::Integer) =
    fold(R, circshift(pad(R, u), mod(k, R.M)))

"product of two ring elements given by their monomial coordinates"
ringmul(R::Ring, u, v) = idft(R, dft(R, u) .* dft(R, v))

emb(R::Ring, u) = R.V * u                                   # tau(u)
trnorm2(R::Ring, u) = sum(abs2, emb(R, u))          # ||u||_Tr^2, u on the MONOMIAL basis
trnorm2s(R::Ring, u) = sum(abs2, R.Vs * u)          # ||u||_Tr^2, u on the DUAL basis

# --- the two noise models -----------------------------------------------------
# They differ by which side of the pivot is spherical in the canonical embedding.
#
#   :spherical  tau(e*) spherical.  The key noise: every RLWE row is drawn as an
#               RLWE* row.  
#
#   :rounding   tau(e) spherical, i.e. white on the M coefficients before the
#               reduction modulo Phi_M.  
#
# The two share their three orbit MEANS, so the diagonal/band/other statistics do
# not separate them; the Frobenius residual does.  The second is measured by
# setting sig_bk = 0, which leaves the arithmetic error alone in the chain.
const MODELS = (:spherical, :rounding)

"the model covariance of e* on the dual basis, normalised to a unit diagonal"
function Gtheory(R::Ring, model::Symbol)
    model === :spherical && return R.toD ./ R.toD[1, 1]
    G = Matrix{Float64}(I, R.N, R.N)
    @inbounds for r in 0:(R.N - 1), rp in 0:(R.N - 1)
        d = mod(rp - r, R.M)
        (d == R.m || d == R.M - R.m) && (G[r + 1, rp + 1] = -0.5)
    end
    return G
end

"diagonal, band and remainder, for the orbits the given model predicts"
function orbit_masks(R::Ring, model::Symbol)
    dia = Matrix(I, R.N, R.N) .== 1
    band = model === :spherical ?
        [mod(r, R.m) == mod(rp, R.m) for r in 0:(R.N-1), rp in 0:(R.N-1)] .& .!dia :
        [(d = mod(rp - r, R.M); d == R.m || d == R.M - R.m)
         for r in 0:(R.N-1), rp in 0:(R.N-1)]
    return dia, band, .!(dia .| band)
end

"the band value the model predicts"
band_target(R::Ring, model::Symbol) = model === :spherical ? -1 / (R.p - 1) : -0.5

"the key noise a model wants: none at all for the rounding regime"
default_sigbk(model::Symbol) = model === :rounding ? 0.0 : 2.0^-26

# --- the N x N linear algebra of the ring, computed once --------------------
const RCACHE = Dict{Tuple{Int,Int,Symbol},Any}()
const RCLOCK = ReentrantLock()
ringcache(R::Ring, what::Symbol, build) =
    lock(RCLOCK) do
        get!(build, RCACHE, (R.p, R.alpha, what))
    end

"the integer matrix of Tr(Omega*_r X^q): Tr(U* e) = <trmat(R) U, e> on Omega"
trmat(R::Ring) = ringcache(R, :trmat, () -> real(transpose(R.V) * R.Vs))::Matrix{Float64}

solveV(R::Ring, y)     = ringcache(R, :luV,     () -> lu(R.V))     \ y
solveVs(R::Ring, y)    = ringcache(R, :luVs,    () -> lu(R.Vs))    \ y
solveStarD(R::Ring, y) = ringcache(R, :lustarD, () -> lu(R.starD)) \ y

# ========================================================== the noise draw ====

"""
    NoiseModel(R)

One convention for every RLWE row of the scheme: h = (Omega*_0)^{-1} h* with h*
spherical in the canonical embedding.  `A` maps N i.i.d. standard normals to the
monomial coordinates of such an h, so that tau(Omega*_0 h) is white.
"""
struct NoiseModel
    A::Matrix{Float64}
end

function NoiseModel(R::Ring)
    U = zeros(ComplexF64, R.N, R.N)
    idx = Dict(d => i for (i, d) in enumerate(R.ds))
    col = 1
    for (i, d) in enumerate(R.ds)
        2d > R.M && continue                      # one representative per pair
        j = idx[R.M - d]
        U[i, col]     = 1 / sqrt(2);  U[j, col]     =  1 / sqrt(2)
        U[i, col + 1] = im / sqrt(2); U[j, col + 1] = -im / sqrt(2)
        col += 2
    end
    return NoiseModel(real(solveV(R, U ./ R.o0)))
end

draw(nz::NoiseModel, sig::Real, rng) = sig .* (nz.A * randn(rng, size(nz.A, 2)))
draw(nz::NoiseModel, sig::Real, rng, k::Int) = sig .* (nz.A * randn(rng, size(nz.A, 2), k))

# ======================================================= test polynomials ====

"v*_f of Proposition 2: dual coordinates f(i/p), constant on each block"
function vstar(R::Ring, fvals::AbstractVector{<:Real})
    a = zeros(Float64, R.N)
    for i in 0:(R.p - 2); a[i * R.m + 1 : (i + 1) * R.m] .= fvals[i + 1]; end
    return a
end

"v_f = (Omega*_0)^{-1} v*_f, monomial coordinates"
vprimal(R::Ring, fvals) = solveStarD(R, vstar(R, fvals))

"w = (1/p) sum_{j<m} Omega_j, monomial coordinates"
function wprimal(R::Ring)
    w = zeros(Float64, R.N); w[1:R.m] .= 1 / R.p; return w
end

"V*_f = sum_{i<=p-2} F_i Omega*_{i m}, dual coordinates"
function Vstar(R::Ring, F::AbstractVector{<:Real})
    a = zeros(Float64, R.N)
    for i in 0:(R.p - 2); a[i * R.m + 1] = F[i + 1]; end
    return a
end

"B*_f = gamma V*_f, dual coordinates; gamma given by its monomial coordinates"
Bstar(R::Ring, F, gamma) = real(solveVs(R, (R.Vs * Vstar(R, F)) .* emb(R, gamma)))

"the rotated polynomial of the multi-value mode: w_frak = w / gamma"
wfrak(R::Ring, gamma) = real(solveV(R, emb(R, wprimal(R)) ./ emb(R, gamma)))

# ============================================================== the scheme ====

"""
    Params(p, alpha; sig_e, sig_bk, Dlog, ell, n)

`n` is the dimension of the INPUT LWE ciphertext.  
"""
struct Params
    R::Ring; p::Int; alpha::Int; n::Int
    sig_e::Float64; sig_bk::Float64
    D::Int; ell::Int; eps::Float64
    nz::NoiseModel
end

function Params(p::Int, alpha::Int; sig_e = 2.0^-12, sig_bk = 2.0^-24,
                Dlog::Int = 7, ell::Int = 5, n::Union{Nothing,Int} = nothing)
    R = Ring(p, alpha)
    return Params(R, p, alpha, n === nothing ? R.N : n, sig_e, sig_bk,
                  2^Dlog, ell, 1 / (2p) - 1 / (2R.M), NoiseModel(R))
end

"""
    Key(z, s)

`z` is the RLWE/ring key, N binary coefficients; `s` is the key of the input
LWE ciphertext, n binary coefficients.  They coincide only when n = N; the
bootstrap is what carries one to the other, the extracted ciphertext being
under the coefficients of z.
"""
struct Key
    z::Vector{Int}
    s::Vector{Int}
end

keygen(P::Params, rng) = Key(rand(rng, 0:1, P.R.N), rand(rng, 0:1, P.n))

# `zhat`, when given, is the spectrum of the ring key, transformed once by
# bootstrap_keys instead of once per row -- 2*ell*n = 6144 times at n = 512,
# ell = 6.  The random stream is untouched, so the keys are bit for bit those
# the same seed produced before.
function rlwe_enc(P::Params, mu::AbstractVector{<:Real}, z, sig, rng; zhat = nothing)
    R = P.R
    a = rand(rng, R.N) .- 0.5
    az = zhat === nothing ? ringmul(R, a, Float64.(z)) : idft(R, dft(R, a) .* zhat)
    b = mod1c(az .+ mu .+ draw(P.nz, sig, rng))
    return (a, b)
end

"RGSW rows: (-msg z / D^t, .) and (msg / D^t, .), t = 1..ell"
function rgsw_enc(P::Params, msg::Real, z, sig, rng; zhat = nothing)
    R = P.R
    rows = [Vector{Tuple{Vector{Float64},Vector{Float64}}}(),
            Vector{Tuple{Vector{Float64},Vector{Float64}}}()]
    zf = Float64.(z)
    for t in 1:P.ell
        Dt = Float64(P.D)^t
        c = zeros(Float64, R.N); c[1] = msg / Dt
        push!(rows[1], rlwe_enc(P, mod1c(-msg .* zf ./ Dt), z, sig, rng; zhat = zhat))
        push!(rows[2], rlwe_enc(P, c, z, sig, rng; zhat = zhat))
    end
    return rows
end

"""
    key_spectra(P, BK) -> Array{ComplexF64,5}

The n RGSW keys once and for all, indexed [frequency, block, level, component,
key], on the M/2+1 kept frequencies.  The frequency comes first so that the
accumulation loop of the external product walks memory contiguously.  The n keys
are transformed in parallel (they write disjoint slices of S).
"""
function key_spectra(P::Params, BK)
    R = P.R
    S = Array{ComplexF64}(undef, nfreq(R), 2, P.ell, 2, P.n)
    Threads.@threads for k in 1:P.n
        for blk in 1:2, t in 1:P.ell, c in 1:2
            S[:, blk, t, c, k] = rdft(R, BK[k][blk][t][c])
        end
    end
    return S
end

"RGSW_z(s_k), k < n; returned as spectra, ready for the blind rotation"
function bootstrap_keys(P::Params, K::Key, rng; spectra::Bool = true)
    zhat = dft(P.R, Float64.(K.z))
    BK = [rgsw_enc(P, Float64(K.s[k]), K.z, P.sig_bk, rng; zhat = zhat) for k in 1:P.n]
    return spectra ? key_spectra(P, BK) : BK
end

"balanced base-D digits of u, levels t = 1..ell"
function decomp(P::Params, u::AbstractVector{<:Real})
    w = round.(Int64, u .* Float64(P.D)^P.ell)
    out = Vector{Vector{Float64}}(undef, P.ell)
    for t in P.ell:-1:1
        r = mod.(w, P.D)
        r = [x > P.D ÷ 2 ? x - P.D : x for x in r]
        out[t] = Float64.(r)
        w = (w .- r) .÷ P.D
    end
    return out
end

"""
    decomp!(Dg, P, u, off)

The same digits, written in place into the columns off+1 .. off+ell of `Dg`, one
coefficient at a time.  
"""
function decomp!(Dg::AbstractMatrix{Float64}, P::Params,
                 u::AbstractVector{<:Real}, off::Int)
    D = P.D; hD = D ÷ 2; sc = Float64(D)^P.ell
    @inbounds for r in eachindex(u)
        w = round(Int64, u[r] * sc)
        for t in P.ell:-1:1
            q = mod(w, D); q > hD && (q -= D)
            Dg[r, off + t] = Float64(q)
            w = (w - q) ÷ D
        end
    end
    return Dg
end

"""
    Workspace(P)

Scratch space for one blind rotation: the padded digit rows, their half
spectrum, the accumulated product, the two real output rows, the multiplier of
the current step, and the two FFTW plans. One workspace per thread.
"""
struct Workspace{PF, PB}
    Zc::Matrix{Float64}          # M x 2ell, the padded digit rows
    Dh::Matrix{ComplexF64}       # (M/2+1) x 2ell, their half spectrum
    Ou::Matrix{ComplexF64}       # (M/2+1) x 2, the accumulated product
    Zr::Matrix{Float64}          # M x 2, the two components back in coefficients
    mu::Vector{ComplexF64}       # (M/2+1), the spectrum of X^{abar_k} - 1
    pf::PF                       # forward plan, real -> half spectrum
    pb::PB                       # backward plan, half spectrum -> real, UNSCALED
end

function Workspace(P::Params)
    R = P.R; h = nfreq(R)
    Zc = zeros(Float64, R.M, 2 * P.ell)     # rows N+1 .. M stay zero for good
    Ou = Matrix{ComplexF64}(undef, h, 2)
    return Workspace(Zc, Matrix{ComplexF64}(undef, h, 2 * P.ell), Ou,
                     Matrix{Float64}(undef, R.M, 2), Vector{ComplexF64}(undef, h),
                     HAVE_FFTW ? plan_rfft(Zc, 1) : nothing,
                     HAVE_FFTW ? plan_brfft(Ou, R.M, 1) : nothing)
end

"""
    ext_prod_core!(P, Ck, mult, acc1, acc2, ws)


The two components are left in `ws.Zr`, on the length-M monomial basis and
already scaled: still to be folded modulo Phi_M.  
"""
function ext_prod_core!(P::Params, Ck, mult, acc1, acc2, ws::Workspace)
    R = P.R; h = nfreq(R)
    decomp!(ws.Zc, P, acc1, 0)
    decomp!(ws.Zc, P, acc2, P.ell)
    if HAVE_FFTW
        mul!(ws.Dh, ws.pf, ws.Zc)
    else
        ws.Dh .= (R.W * ws.Zc)[1:h, :]
    end
    fill!(ws.Ou, 0)
    @inbounds for c in 1:2, t in 1:P.ell
        for f in 1:h
            ws.Ou[f, c] += ws.Dh[f, t] * Ck[f, 1, t, c] + ws.Dh[f, P.ell + t] * Ck[f, 2, t, c]
        end
    end
    @inbounds for c in 1:2, f in 1:h
        ws.Ou[f, c] *= mult[f]
    end
    if HAVE_FFTW
        mul!(ws.Zr, ws.pb, ws.Ou)               # brfft is unnormalised
        ws.Zr .*= inv(R.M)
    else
        ws.Zr .= real(R.Wi * unhalf(R, ws.Ou))  # R.Wi already carries the 1/M
    end
    return nothing
end

"""
    ext_prod_add!(P, Ck, mult, acc1, acc2, ws)

The step of the blind rotation: acc <- mod1(acc + product).  The reduction
modulo Phi_M, the addition into the accumulator and the reduction modulo 1 are
fused into a single pass over the N coefficients, so that the whole step
allocates nothing at all.
"""
function ext_prod_add!(P::Params, Ck, mult, acc1, acc2, ws::Workspace)
    R = P.R
    ext_prod_core!(P, Ck, mult, acc1, acc2, ws)
    @inbounds for i in 0:(R.p - 2), j in 0:(R.m - 1)
        r = i * R.m + j + 1; q = R.N + j + 1
        x = acc1[r] + (ws.Zr[r, 1] - ws.Zr[q, 1]); acc1[r] = x - round(x)
        y = acc2[r] + (ws.Zr[r, 2] - ws.Zr[q, 2]); acc2[r] = y - round(y)
    end
    return nothing
end

"the product alone, as a fresh pair -- for one-off calls and for test R5"
function ext_prod_spec!(P::Params, Ck, mult, acc, ws::Workspace)
    ext_prod_core!(P, Ck, mult, acc[1], acc[2], ws)
    z = fold(P.R, ws.Zr)
    return (mod1c(z[:, 1]), mod1c(z[:, 2]))
end

ext_prod_spec(P::Params, Ck, mult, acc) =
    ext_prod_spec!(P, Ck, mult, acc, Workspace(P))

"the same external product from keys in coefficient form -- the reference route"
function ext_prod(P::Params, C, acc)
    R = P.R
    da = decomp(P, acc[1]); db = decomp(P, acc[2])
    na = zeros(Float64, R.N); nb = zeros(Float64, R.N)
    for t in 1:P.ell
        na .+= ringmul(R, da[t], C[1][t][1]) .+ ringmul(R, db[t], C[2][t][1])
        nb .+= ringmul(R, da[t], C[1][t][2]) .+ ringmul(R, db[t], C[2][t][2])
    end
    return (mod1c(na), mod1c(nb))
end

"""
    blind_rotate(P, a, b, K, S, v0) -> (acc, imath)

The CGGI loop, acc <- acc + ((X^{abar_k} - 1) RGSW(s_k)) [x] acc, rotating the
PRIMAL v0.  `imath` is the exponent the chain has realised, so that the
noiseless accumulator is exactly X^{imath} v0.
"""
function blind_rotate(P::Params, a, b, K::Key, S, v0; ws::Workspace = Workspace(P))
    R = P.R; h = nfreq(R)
    abar = round.(Int, R.M .* a)
    a0 = round(Int, -R.M * b - R.M * P.eps)
    acc1 = zeros(Float64, R.N)                  # the two accumulator rows, held
    acc2 = xpow(R, v0, mod(a0, R.M))            # in place for the whole chain
    for k in 1:P.n
        kk = mod(abar[k], R.M)
        @inbounds for f in 1:h                       # spectrum of X^{abar_k} - 1
            ws.mu[f] = cis(-2pi * kk * (f - 1) / R.M) - 1
        end
        ext_prod_add!(P, view(S, :, :, :, :, k), ws.mu, acc1, acc2, ws)
    end
    imath = mod(a0 + sum(abar .* K.s), R.M)
    return (acc1, acc2), imath
end

# ============================================================== extraction ====

"""
    extract(P, U, acc)

LWE_z of Tr(U* . phase(acc)), the functional being carried by the integer
coefficients c_q = Tr(U* X^q).
"""
function extract_coeffs(R::Ring, U::AbstractVector{<:Real})
    tau = R.Vs * U
    T = zeros(ComplexF64, R.M)
    @inbounds for (i, d) in enumerate(R.ds); T[d + 1] = tau[i]; end
    # c_q = sum_d tau_d exp(2i pi d q / M) is M times an inverse transform of tau
    # placed on the units.
    return real(R.M .* (HAVE_FFTW ? ifft(T) : R.Wi * T))
end

function extract(P::Params, U::AbstractVector{<:Real}, acc;
                 c = extract_coeffs(P.R, U))
    R = P.R
    a, b = acc
    ahat = Vector{Float64}(undef, R.N)
    @inbounds for j in 0:(R.N - 1)          # j + r < 2N - 1 < 2M, so one wrap
        s = 0.0
        for r in 0:(R.N - 1)
            q = j + r; q >= R.M && (q -= R.M)
            s += c[q + 1] * a[r + 1]
        end
        ahat[j + 1] = s
    end
    bh = 0.0
    @inbounds for r in 1:R.N; bh += c[r] * b[r]; end
    return ahat, bh
end

"the extracted LWE is under the coefficients of the ring key z"
lwe_phase(ahat, bhat, K::Key) = mod1c(bhat - dot(ahat, K.z))

"U* = Omega*_0, dual coordinates"
function pivotfun(R::Ring)
    u = zeros(Float64, R.N); u[1] = 1.0; return u
end

# ========================================================== the three modes ====

function gamma_one(R::Ring)
    g = zeros(Float64, R.N); g[1] = 1.0; return g
end

"gamma = M Omega*_0 = 1 - X^N, monomial coordinates"
function gamma_pivot(R::Ring)
    z = zeros(Float64, R.M); z[1] = 1.0; z[R.N + 1] -= 1.0; return fold(R, z)
end

const MODES = ("SVM", "MVM1", "MVMpiv")

"the polynomial the mode rotates"
function rotated(P::Params, mode::AbstractString, fvals, F)
    R = P.R
    mode == "SVM"    && return vprimal(R, fvals)
    mode == "MVM1"   && return wfrak(R, gamma_one(R))
    mode == "MVMpiv" && return wfrak(R, gamma_pivot(R))
    error("unknown mode $mode")
end

"the functional U* the mode extracts with, dual coordinates"
function functional(P::Params, mode::AbstractString, F)
    R = P.R
    mode == "SVM"    && return pivotfun(R)
    mode == "MVM1"   && return Bstar(R, F, gamma_one(R))
    mode == "MVMpiv" && return Bstar(R, F, gamma_pivot(R))
    error("unknown mode $mode")
end

function bootstrap(P::Params, mode, a, b, K::Key, S, fvals, F;
                   ws::Workspace = Workspace(P), U = functional(P, mode, F),
                   c = extract_coeffs(P.R, U), v0 = rotated(P, mode, fvals, F))
    acc, im = blind_rotate(P, a, b, K, S, v0; ws = ws)
    return acc, im, extract(P, U, acc; c = c)
end

# ========================================== the five named maps of Figure 3 ====

const MAPS = ("identity", "sign", "rectifier", "square", "comparison")

"F_i = p f(i/p), i <= p-2"
function lift(p::Int, name::AbstractString)
    r = [mods(i, p) for i in 0:(p - 2)]
    name == "identity"   && return Float64.(r)
    name == "sign"       && return Float64.(sign.(r))
    name == "rectifier"  && return Float64.(max.(r, 0))
    name == "square"     && return Float64[mods(i * i, p) for i in 0:(p - 2)]
    name == "comparison" && return Float64[x > 0 ? 1.0 : 0.0 for x in r]
    error("unknown map $name")
end

"B_f = gamma V_f in R, with V_f = (Omega*_0)^{-1} V*_f"
cofactor(R::Ring, F, gamma) = ringmul(R, gamma, solveStarD(R, Vstar(R, F)))

"sigma(E_2)/sigma(E_1) = ||B_f||_Tr / sqrt(N), the spherical model"
function predicted_amp(R::Ring, F, gamma; model::Symbol = :spherical)
    model === :spherical && return sqrt(trnorm2(R, cofactor(R, F, gamma)) / R.N)
    # Corollary 2: the form is the trace norm of B*_f = gamma V*_f in the dual, and
    # sigma^2(E_1) is its value at B*_f = Omega*_0, namely ||Omega*_0||^2 = 2/M.
    return sqrt(trnorm2s(R, Bstar(R, F, gamma)) / trnorm2s(R, pivotfun(R)))
end

# ============================================================= experiments ====

"a fresh LWE input: mask A and body b, on the M-grid"
function rand_input(P::Params, rng, K::Key; aligned::Bool = true)
    i = rand(rng, 0:(P.p - 1)); mu = mods(i, P.p) / P.p
    A = aligned ? mod1c(rand(rng, 0:(P.R.M - 1), P.n) ./ P.R.M) : rand(rng, P.n) .- 0.5
    b = mod1c(dot(A, K.s) + mu + P.sig_e * randn(rng))
    return A, b
end

"""
    noiseless_out(P, mode, imath, fvals, F)

Tr(U* . X^imath v_0): what the chain would output with no noise at all.  
"""
function noiseless_out(P::Params, mode, imath::Integer, fvals, F)
    R = P.R
    U = functional(P, mode, F); v0 = rotated(P, mode, fvals, F)
    return mod1c(real(sum((R.Vs * U) .* emb(R, xpow(R, v0, imath)))))
end

"""
    chunkup(nb)

The draws 1:nb split into one contiguous block per thread.  The parallel loops
below run over the blocks rather than over the draws, so that each block builds
exactly one Workspace and no iteration ever has to ask which thread it is on.
"""
chunkup(nb::Int) = collect(Iterators.partition(1:nb, max(1, cld(nb, Threads.nthreads()))))

"""
    output_errors(P, mode, nb, rng, K, S, fvals, F)

The output error of one mode, nb independent bootstraps.

The nb inputs are drawn FIRST, sequentially, from the single `rng`; only the
rotations, which do not touch it and do not touch each other, are then run in
parallel.  
"""
function output_errors(P::Params, mode, nb::Int, rng, K::Key, S, fvals, F;
                       aligned::Bool = true)
    out = Vector{Float64}(undef, nb)
    ins = [rand_input(P, rng, K; aligned = aligned) for _ in 1:nb]
    U = functional(P, mode, F); cq = extract_coeffs(P.R, U)   # constant, hoisted
    v0 = rotated(P, mode, fvals, F)
    blk = chunkup(nb)
    Threads.@threads for j in eachindex(blk)
        ws = Workspace(P)
        for k in blk[j]
            A, b = ins[k]
            _, im, (ah, bh) = bootstrap(P, mode, A, b, K, S, fvals, F;
                                        ws = ws, U = U, c = cq, v0 = v0)
            out[k] = mod1c(lwe_phase(ah, bh, K) - noiseless_out(P, mode, im, fvals, F))
        end
    end
    return out
end

"""
    accumulator_errors(P, mode, nb, rng, K, S, fvals, F)

e (monomial coordinates) and e* = Omega*_0 e (dual coordinates), nb rotations.
This is the loop that carries the whole cost of Figure 3 and of Table 1, and it
is parallelised as in `output_errors`: inputs drawn sequentially, rotations run
in parallel, sample independent of the thread count.
"""
function accumulator_errors(P::Params, mode, nb::Int, rng, K::Key, S, fvals, F)
    R = P.R
    E = Matrix{Float64}(undef, nb, R.N)
    v0 = rotated(P, mode, fvals, F)
    ins = [rand_input(P, rng, K) for _ in 1:nb]
    blk = chunkup(nb)
    Threads.@threads for c in eachindex(blk)
        ws = Workspace(P)
        for k in blk[c]
            A, b = ins[k]
            acc, im = blind_rotate(P, A, b, K, S, v0; ws = ws)
            ph = mod1c(acc[2] .- ringmul(R, acc[1], Float64.(K.z)))
            E[k, :] = mod1c(ph .- xpow(R, v0, im))
        end
    end
    return E, E * transpose(R.starD)
end

"the real vector w such that Tr(U* . e) = <w, e> for e on the monomial basis"
functional_vector(R::Ring, U) = trmat(R) * U

"""
    exp_amplification(p, alpha; ...)

Amplification sigma(E_2)/sigma(E_1) for the five named maps and the two designs
gamma = 1 and gamma = M Omega*_0, against the prediction ||B_f||_Tr / sqrt(N)
of the spherical model.  This is Figure 3 of the paper.

"""
function exp_amplification(p::Int, alpha::Int; nb::Int = 8000, n::Int = 256,
                           model::Symbol = :spherical, sig_bk = nothing,
                           ell::Int = 6, Dlog::Int = 7,
                           seed::Int = 0, maps = MAPS, verbose::Bool = true)
    @assert model in MODELS "unknown model $model"
    sig_bk = sig_bk === nothing ? default_sigbk(model) : sig_bk
    model === :rounding && sig_bk != 0.0 &&
        @warn "model :rounding with sig_bk = $sig_bk measures a mixture of the two regimes"
    P = Params(p, alpha; sig_e = 0.0, sig_bk = sig_bk, Dlog = Dlog, ell = ell, n = n)
    R = P.R
    rng = MersenneTwister(seed)
    K = keygen(P, rng); S = bootstrap_keys(P, K, rng)
    F0 = lift(p, "identity")
    E, _ = accumulator_errors(P, "MVM1", nb, MersenneTwister(seed + 1), K, S, F0 ./ p, F0)
    e1 = E * functional_vector(R, pivotfun(R))          # SVM : U* = Omega*_0
    s1 = std(e1)
    gam = Dict("MVM1" => gamma_one(R), "MVMpiv" => gamma_pivot(R))
    rad = 1 / (2p)                       # the decoding radius: |E| < rad decrypts
    rows = NamedTuple[]
    for nm in maps
        F = lift(p, nm)
        for mode in ("MVM1", "MVMpiv")
            e2 = E * functional_vector(R, Bstar(R, F, gam[mode]))
            s2 = std(e2)
            # Proposition 3 makes the bootstrap output error exactly this linear
            # functional of the accumulator error (test T2 checks it against the
            # full chain to machine precision), so the draws beyond the decoding
            # radius are the decryption failures, counted without re-rotating.
            push!(rows, (p = p, alpha = alpha, N = R.N, n = n, nb = nb, map = nm,
                         mode = mode, model = model, sigma1 = s1, sigma2 = s2,
                         measured = s2 / s1,
                         predicted = predicted_amp(R, F, gam[mode]; model = model),
                         radius = rad, margin = rad / s2,
                         worst = maximum(abs, e2), fails = count(>(rad), abs.(e2))))
            verbose && @printf("  p=%-3d a=%d %-9s %-11s %-7s meas %8.3f  pred %8.3f  ratio %.4f   margin %5.1f sd  fails %d/%d\n",
                               p, alpha, String(model), nm, mode, rows[end].measured,
                               rows[end].predicted, rows[end].measured / rows[end].predicted,
                               rows[end].margin, rows[end].fails, nb)
        end
    end
    #
    emax = maximum(abs, E)
    emax > 0.45 && @warn @sprintf("the accumulator coordinates reach %.3f of the 1/2 wrap (sig_bk = %.3g): they are folding and every amplification is biased downwards. Lower sig_bk -- 2.0^-30 keeps the maximum near %.2g at N = 2058, n = 512.", emax, sig_bk, 0.05)
    verbose && @printf("  accumulator: sd %.3g, max %.3g of the 1/2 wrap\n", std(E), emax)
    return rows
end

"""
    bench(p, alpha; n, ell, nb)

Where the time goes, and how it scales.  
"""
function bench(p::Int, alpha::Int; n::Int = 512, ell::Int = 6, nb::Int = 32,
               sig_bk = 2.0^-30)
    ta = @timed begin
        P = Params(p, alpha; sig_e = 0.0, sig_bk = sig_bk, Dlog = 7, ell = ell, n = n)
        F = lift(p, "identity")
        trmat(P.R); Bstar(P.R, F, gamma_one(P.R)); wfrak(P.R, gamma_one(P.R))
        P
    end
    P = ta.value; R = P.R; F = lift(p, "identity"); f = F ./ p
    rng = MersenneTwister(11); K = keygen(P, rng)
    t0 = @timed bootstrap_keys(P, K, rng)
    S = t0.value
    accumulator_errors(P, "MVM1", 1, MersenneTwister(1), K, S, f, F)        # warm-up
    t1 = @timed accumulator_errors(P, "MVM1", nb, MersenneTwister(2), K, S, f, F)
    @printf("  p=%d alpha=%d  N=%d  n=%d  ell=%d   threads %d   FFTW %s\n",
            p, alpha, R.N, n, ell, Threads.nthreads(), HAVE_FFTW ? "yes" : "no")
    @printf("  ring N x N  %7.2f s  %7.2f GiB   (once per ring, then cached)\n",
            ta.time, ta.bytes / 2^30)
    @printf("  key setup   %7.2f s  %7.2f GiB\n", t0.time, t0.bytes / 2^30)
    @printf("  %4d rot.   %7.2f s  %7.2f GiB   ->  %6.1f ms  and  %5.1f MiB  per rotation\n",
            nb, t1.time, t1.bytes / 2^30, 1000 * t1.time / nb, t1.bytes / nb / 2^20)
    return (ring = ta.time, setup = t0.time,
            per_rotation = t1.time / nb, threads = Threads.nthreads())
end

"""
    calibrate_sigbk(p, alpha; n, ell, target_sd, ...)

The key noise that leaves the WORST of the ten amplifications `target_sd`
standard deviations inside the decoding radius 1/(2p), so that the setting is
one that actually decrypts.

"""
function calibrate_sigbk(p::Int, alpha::Int; n::Int = 512, ell::Int = 6,
                         Dlog::Int = 7, target_sd::Real = 8.0, ref = 2.0^-30,
                         nb::Int = 64, seed::Int = 11)
    local R, s1, emax
    #
    for attempt in 1:12
        P = Params(p, alpha; sig_e = 0.0, sig_bk = ref, Dlog = Dlog, ell = ell, n = n)
        R = P.R
        rng = MersenneTwister(seed); K = keygen(P, rng); S = bootstrap_keys(P, K, rng)
        F0 = lift(p, "identity")
        E, _ = accumulator_errors(P, "MVM1", nb, MersenneTwister(seed + 1), K, S,
                                  F0 ./ p, F0)
        emax = maximum(abs, E)
        s1 = std(E * functional_vector(R, pivotfun(R)))
        emax < 0.05 && break
        attempt == 12 && error("calibration cannot get clear of the torus wrap")
        ref /= 64.0
    end
    amax = maximum(predicted_amp(R, lift(p, nm), g)
                   for nm in MAPS, g in (gamma_one(R), gamma_pivot(R)))
    want = ref * (1 / (2p)) / (target_sd * amax * s1)
    return (sig_bk = 2.0^floor(log2(want)), amax = amax,
            sigma1_ref = s1, ref = ref, emax_ref = emax)
end

"""
    exp_decrypt(p, alpha; sig_bk, ...)

The end-to-end check: nb complete bootstraps per mode, counting how many land outside the
decoding radius 1/(2p).  Proposition 3 gives the output error as a
functional of the accumulator, and `exp_amplification` counts the failures that
way; this runs the whole chain instead.
"""
function exp_decrypt(p::Int, alpha::Int; sig_bk, n::Int = 512, ell::Int = 6,
                     Dlog::Int = 7, nb::Int = 200, seed::Int = 11,
                     mp = "identity", verbose::Bool = true)
    P = Params(p, alpha; sig_e = 0.0, sig_bk = sig_bk, Dlog = Dlog, ell = ell, n = n)
    rng = MersenneTwister(seed); K = keygen(P, rng); S = bootstrap_keys(P, K, rng)
    F = lift(p, mp); f = F ./ p; rad = 1 / (2p)
    out = NamedTuple[]
    for mode in MODES
        e = output_errors(P, mode, nb, MersenneTwister(seed + 2), K, S, f, F)
        push!(out, (p = p, alpha = alpha, mode = mode, map = mp, nb = nb,
                    sigma = std(e), radius = rad, worst = maximum(abs, e),
                    margin = rad / std(e), fails = count(>(rad), abs.(e))))
        verbose && @printf("  p=%-3d a=%d %-7s %-10s  sd %9.3g  worst %9.3g  radius %7.4f  margin %5.1f sd  fails %d/%d\n",
                           p, alpha, mode, mp, out[end].sigma, out[end].worst,
                           rad, out[end].margin, out[end].fails, nb)
    end
    return out
end

"the three orbits of the spherical model: diagonal, r = r' mod m, other"
function orbit_masks(R::Ring)
    same = [mod(r, R.m) == mod(rp, R.m) for r in 0:(R.N-1), rp in 0:(R.N-1)]
    dia = Matrix(I, R.N, R.N) .== 1
    return dia, same .& .!dia, .!same
end

"""
    exp_covariance(p, alpha; ...)

Empirical covariance G^ of the N dual coordinates of e* = Omega*_0 e, against
the spherical model G = sigma_c^2 (M I_N - m Pi) = sigma_c^2 Gr(Omega) and
against the white model G = sigma^2 I_N.  Key regime: sigma_BK > 0 with the
gadget precision pushed out of the way.  
"""
function exp_covariance(p::Int, alpha::Int; nb::Int = 3000, n::Int = 128,
                        model::Symbol = :spherical, sig_bk = nothing,
                        ell::Int = 5, seed::Int = 0,
                        mode = "MVM1", mp = "identity", verbose::Bool = true)
    @assert model in MODELS "unknown model $model"
    sig_bk = sig_bk === nothing ? default_sigbk(model) : sig_bk
    P = Params(p, alpha; sig_e = 0.0, sig_bk = sig_bk, Dlog = 7, ell = ell, n = n)
    R = P.R
    rng = MersenneTwister(seed)
    K = keygen(P, rng); S = bootstrap_keys(P, K, rng)
    F = lift(p, mp); f = F ./ p
    Em, Es = accumulator_errors(P, mode, nb, MersenneTwister(seed + 1), K, S, f, F)
    emax = maximum(abs, Em)      # the wrap of mod1c would bias G, see exp_amplification
    emax > 0.45 && @warn @sprintf("the accumulator reaches %.3f of the 1/2 wrap at sig_bk = %.3g: G is biased", emax, sig_bk)
    G = transpose(Es) * Es ./ nb
    Gn = G ./ G[1, 1]
    Gth = Gtheory(R, model)
    Wh = Matrix{Float64}(I, R.N, R.N)
    rel(A, B) = norm(A - B) / norm(B)
    h = nb ÷ 2
    G1 = transpose(Es[1:h, :]) * Es[1:h, :] ./ h
    G2 = transpose(Es[h+1:end, :]) * Es[h+1:end, :] ./ (nb - h)
    dia, band, other = orbit_masks(R, model)
    out = (p = p, alpha = alpha, N = R.N, nb = nb, n = n, mode = mode, model = model,
           diag = mean(Gn[dia]), band = mean(Gn[band]),
           other = any(other) ? mean(Gn[other]) : NaN,
           target = band_target(R, model),
           res_spherical = rel(Gn, Gth), res_white = rel(Gn, Wh),
           floor = rel(G1 ./ G1[1, 1], G2 ./ G2[1, 1]), emax = emax,
           G = Gn, Gth = Gth)
    verbose && @printf("  p=%-3d a=%d %-9s N=%-4d nb=%-6d diag %7.4f  band %8.4f  other %8.4f  (model %8.4f)  res %.3f / white %.3f / floor %.3f\n",
                       p, alpha, String(model), R.N, nb, out.diag, out.band, out.other, out.target,
                       out.res_spherical, out.res_white, out.floor)
    return out
end

# ======================================================== table and figures ====

"Table 1 of the paper, as text"
function table1(rows)
    h = "  p  a    N      K  model      diag     band    other    target   model  white   floor"
    lines = [h, "  " * repeat("-", length(h) - 2)]
    for r in rows
        push!(lines, @sprintf("%3d %2d %4d %7d  %-9s %8.4f %8.4f %8.4f %9.4f  %6.3f %6.3f %6.3f",
                              r.p, r.alpha, r.N, r.nb, String(r.model), r.diag, r.band,
                              r.other, r.target, r.res_spherical, r.res_white, r.floor))
    end
    return join(lines, "\n")
end

"the same table as a LaTeX tabular, ready to paste into the paper"
function table1_tex(rows)
    b = IOBuffer()
    println(b, "\\begin{tabular}{cclrrrrrrrrr}")
    println(b, "\\hline")
    println(b, "& & & & & \\multicolumn{3}{c}{\$\\hat G/\\hat G_{00}\$} & &")
    println(b, "\\multicolumn{3}{c}{relative residual} \\\\")
    println(b, "\\cline{6-8}\\cline{10-12}")
    println(b, "\$p\$ & \$\\alpha\$ & \$N\$ & \$K\$ & model & diag. & band & other & target")
    println(b, " & model & white & floor \\\\")
    println(b, "\\hline")
    for r in rows
        @printf(b, "\$%d\$ & \$%d\$ & \$%d\$ & \$%d\$ & %s & \$%.4f\$ & \$%.4f\$ & \$%.4f\$ & \$%.4f\$ & \$%.3f\$ & \$%.3f\$ & \$%.3f\$ \\\\\n",
                r.p, r.alpha, r.N, r.nb, String(r.model), r.diag, r.band, r.other,
                r.target, r.res_spherical, r.res_white, r.floor)
    end
    println(b, "\\hline")
    println(b, "\\end{tabular}")
    return String(take!(b))
end

"CSV fallback, so that every number survives even without a plotting backend"
function write_csv(amp, cov, prefix)
    open(prefix * "_amp.csv", "w") do io
        println(io, "p,alpha,N,n,nb,map,mode,sigma1,sigma2,measured,predicted")
        for r in amp
            @printf(io, "%d,%d,%d,%d,%d,%s,%s,%.6e,%.6e,%.6f,%.6f\n",
                    r.p, r.alpha, r.N, r.n, r.nb, r.map, r.mode,
                    r.sigma1, r.sigma2, r.measured, r.predicted)
        end
    end
    open(prefix * "_cov.csv", "w") do io
        println(io, "p,alpha,N,nb,diag,band,other,target,res_spherical,res_white,floor")
        for r in cov
            @printf(io, "%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    r.p, r.alpha, r.N, r.nb, r.diag, r.band, r.other, r.target,
                    r.res_spherical, r.res_white, r.floor)
        end
    end
    open(prefix * "_gram.csv", "w") do io       # the cloud, one row per entry
        println(io, "p,alpha,r,rprime,orbit,measured,model")
        for c in cov
            R = Ring(c.p, c.alpha)
            for r in 1:c.N, rp in 1:c.N
                orb = r == rp ? "diag" :
                      (mod(r - 1, R.m) == mod(rp - 1, R.m) ? "band" : "other")
                @printf(io, "%d,%d,%d,%d,%s,%.6e,%.6e\n",
                        c.p, c.alpha, r - 1, rp - 1, orb, c.G[r, rp], c.Gth[r, rp])
            end
        end
    end
    return (prefix * "_amp.csv", prefix * "_cov.csv", prefix * "_gram.csv")
end

const HAVE_PLOTS = try
    @eval using Plots
    true
catch
    false
end

"""
    figures(amp, cov; prefix)

Three figures:
  `<prefix>_funcs.pdf`  the amplification of the two designs on the five maps,
                        predicted (bars) against measured (dots) -- Figure 3;
  `<prefix>_cov.pdf`    the measured accumulator covariance, its band against
                        the two models, and the amplification parity plot;
  `<prefix>_cloud.pdf`  the covariance as a cloud: every entry of G^ against
                        the entry the spherical model predicts, coloured by
                        orbit -- three clusters and nothing in between is the
                        whole content of (eq:Gcan).
Falls back to CSV when Plots.jl is not installed.
"""
function figures(amp, cov; prefix::AbstractString = "mvm_jl")
    files = write_csv(amp, cov, prefix)
    if !HAVE_PLOTS
        @info "Plots.jl not found -- the numbers were written as CSV instead" files
        return files
    end
    gr()

    # ------------------------------------------------------------- Figure 3
    settings = sort(unique([(r.p, r.alpha) for r in amp]))
    panels = Any[]
    for (p, a) in settings
        sel = [r for r in amp if r.p == p && r.alpha == a]
        ms = [m for m in MAPS if any(r -> r.map == m, sel)]
        x = collect(1:length(ms)); wdt = 0.38
        pl = plot(; xticks = (x, ms), xrotation = 20,
                  ylabel = "sigma(E2)/sigma(E1)",
                  title = "p = $p, alpha = $a  (N = $(sel[1].N))",
                  legend = :topleft, grid = true)
        for (k, (mode, lab, col)) in enumerate((("MVM1", "gamma = 1", :steelblue),
                                                ("MVMpiv", "gamma = M Omega*_0", :indianred)))
            off = (k - 1.5) * wdt
            pr = [first(r.predicted for r in sel if r.map == m && r.mode == mode) for m in ms]
            me = [first(r.measured  for r in sel if r.map == m && r.mode == mode) for m in ms]
            bar!(pl, x .+ off, pr; bar_width = wdt, color = col, alpha = 0.55,
                 label = lab * " predicted")
            scatter!(pl, x .+ off, me; color = col, markerstrokecolor = :black,
                     markersize = 5, label = lab * " measured")
        end
        push!(panels, pl)
    end
    f1 = plot(panels...; layout = (1, length(panels)),
              size = (620 * length(panels), 420))
    savefig(f1, prefix * "_funcs.pdf"); savefig(f1, prefix * "_funcs.png")

    # ------------------------------------------------------------- Figure 4
    # panel (a) shows the band structure, so pick the setting with the largest
    # m = M/p: at alpha = 1 the band is the whole off-diagonal and there is
    # nothing to see.
    ref = cov[argmax([Ring(c.p, c.alpha).m for c in cov])]
    v = maximum(abs, ref.G)
    a1 = heatmap(0:(ref.N-1), 0:(ref.N-1), ref.G; c = :RdBu, clims = (-v, v),
                 yflip = true, xlabel = "r'", ylabel = "r",
                 title = "(a) Ghat/Ghat00, p = $(ref.p), alpha = $(ref.alpha), K = $(ref.nb)")
    ps = [r.p for r in cov]
    a2 = scatter(ps, [r.band for r in cov]; label = "measured band r = r' mod m",
                 xlabel = "p", ylabel = "Ghat_rr' / Ghat_00",
                 title = "(b) the band, against the two models", grid = true)
    pp = range(minimum(ps) * 0.9, maximum(ps) * 1.05; length = 200)
    plot!(a2, pp, -1 ./ (pp .- 1); label = "-1/(p-1)  (spherical)", lw = 1.4)
    hline!(a2, [0.0]; ls = :dash, color = :black, label = "white model")
    pr = [r.predicted for r in amp]; me = [r.measured for r in amp]
    lo = 0.8 * min(minimum(pr), minimum(me)); hi = 1.25 * max(maximum(pr), maximum(me))
    a3 = scatter(pr, me; xscale = :log10, yscale = :log10, markersize = 4,
                 markerstrokecolor = :black, legend = false,
                 xlabel = "predicted", ylabel = "measured",
                 title = "(c) amplification, $(length(pr)) configurations", grid = true)
    plot!(a3, [lo, hi], [lo, hi]; color = :black, lw = 1)
    f2 = plot(a1, a2, a3; layout = (1, 3), size = (1500, 430))
    savefig(f2, prefix * "_cov.pdf"); savefig(f2, prefix * "_cov.png")

    # --------------------------------------------------- the covariance cloud
    f3 = plot(; xlabel = "model  G_rr' / G_00", ylabel = "measured  Ghat_rr' / Ghat_00",
              title = "the accumulator covariance as a cloud", grid = true,
              legend = :topleft, size = (620, 560))
    for (orb, col) in (("other", :grey), ("band", :indianred), ("diag", :steelblue))
        xs = Float64[]; ys = Float64[]
        for c in cov
            R = Ring(c.p, c.alpha)
            for r in 1:c.N, rp in 1:c.N
                o = r == rp ? "diag" :
                    (mod(r - 1, R.m) == mod(rp - 1, R.m) ? "band" : "other")
                o == orb || continue
                push!(xs, c.Gth[r, rp]); push!(ys, c.G[r, rp])
            end
        end
        isempty(xs) && continue
        scatter!(f3, xs, ys; color = col, markersize = 2.2, markerstrokewidth = 0,
                 alpha = 0.45, label = orb)
    end
    lo = minimum(minimum(c.Gth) for c in cov) - 0.15
    plot!(f3, [lo, 1.15], [lo, 1.15]; color = :black, lw = 1, label = "y = x")
    savefig(f3, prefix * "_cloud.pdf"); savefig(f3, prefix * "_cloud.png")

    return (prefix * "_funcs.pdf", prefix * "_cov.pdf", prefix * "_cloud.pdf", files...)
end

# ================================================================== tests ====

"""
    selftest(p, alpha)

Twenty-two checks, in five groups.

  R0..R5  the ring and the factorization: products against the embeddings,
          v_f in (1/p)R, w_frak in (1/p^2)R for the pivot, v*_f = V*_f w, the
          invariance of B*_f w_frak under gamma, and the batched external
          product against the reference one;
  T2      Proposition 3: the output error of a full bootstrap is exactly the
          linear functional Tr(U* e) of the accumulator error that run
          produced -- which is what licenses measuring the amplification on
          accumulator draws rather than on whole bootstraps;
  N0      the key noise law: tau(Omega*_0 h) is white;
  T1a     the whole pipeline, with a zero mask, is exact -- no rounding of any
          kind is then involved and the three modes must return the message to
          machine precision;
  T1b     the gadget decomposition itself: remainder within D^{-ell}/2 in the
          torus, and digits within [-D/2, D/2];
  T1c     its chain-level consequence, as a one-sided bound.
"""
function selftest(p::Int, alpha::Int; verbose::Bool = true)
    P = Params(p, alpha; sig_e = 0.0, sig_bk = 0.0, Dlog = 7, ell = 8)
    R = P.R
    rng = MersenneTwister(1)
    K = keygen(P, rng); S = bootstrap_keys(P, K, rng)
    f = Float64[mods(i, p) / p for i in 0:(p - 2)]
    F = Float64[mods(i, p)     for i in 0:(p - 2)]
    res = Tuple{String,String,Float64,Float64}[]

    u = randn(rng, R.N); v = randn(rng, R.N)
    push!(res, ("R0", "ring product vs embeddings",
                maximum(abs.(emb(R, ringmul(R, u, v)) .- emb(R, u) .* emb(R, v))), 1e-9))
    vf = vprimal(R, f)
    push!(res, ("R1", "v_f in (1/p)R", maximum(abs.(p .* vf .- round.(p .* vf))), 1e-9))
    wp = wfrak(R, gamma_pivot(R))
    push!(res, ("R2", "w_frak(pivot) in (1/p^2)R",
                maximum(abs.(p^2 .* wp .- round.(p^2 .* wp))), 1e-8))
    prod = real(solveVs(R, (R.Vs * Vstar(R, F)) .* emb(R, wprimal(R))))
    push!(res, ("R3", "v*_f = V*_f . w", maximum(abs.(prod .- vstar(R, f))), 1e-9))
    for (nm, g) in (("1", gamma_one(R)), ("pivot", gamma_pivot(R)))
        q = real(solveVs(R, (R.Vs * Bstar(R, F, g)) .* emb(R, wfrak(R, g))))
        push!(res, ("R4", "B*_f . w_frak invariant, gamma=$nm",
                    maximum(abs.(q .- vstar(R, f))), 1e-8))
    end
    BKc = bootstrap_keys(P, K, MersenneTwister(9); spectra = false)
    S9 = key_spectra(P, BKc)
    accr = (rand(rng, R.N) .- 0.5, rand(rng, R.N) .- 0.5)
    kk = rand(rng, 0:(R.M - 1))
    Cshift = [[(xpow(R, x, kk) .- x, xpow(R, y, kk) .- y) for (x, y) in BKc[1][blk]]
              for blk in 1:2]
    ref = ext_prod(P, Cshift, accr)
    bat = ext_prod_spec(P, view(S9, :, :, :, :, 1), shiftspec(R, kk) .- 1, accr)
    push!(res, ("R5", "batched external product",
                max(maximum(abs.(ref[1] .- bat[1])), maximum(abs.(ref[2] .- bat[2]))), 1e-10))

    # T2 : Proposition 3 -- the output error of a full bootstrap is exactly the
    # linear functional Tr(U* e) of the accumulator error it produced.
    P2 = Params(p, alpha; sig_e = 0.0, sig_bk = 2.0^-24, Dlog = 7, ell = 6)
    K2 = keygen(P2, MersenneTwister(4))
    S2 = bootstrap_keys(P2, K2, MersenneTwister(4))
    for mode in MODES
        v0 = rotated(P2, mode, f, F); U = functional(P2, mode, F)
        A = mod1c(rand(MersenneTwister(6), 0:(R.M - 1), P2.n) ./ R.M)
        b = mod1c(dot(A, K2.s) + mods(1, p) / p)
        acc, im, (ah, bh) = bootstrap(P2, mode, A, b, K2, S2, f, F)
        eacc = mod1c(mod1c(acc[2] .- ringmul(R, acc[1], Float64.(K2.z))) .- xpow(R, v0, im))
        got = mod1c(lwe_phase(ah, bh, K2) - noiseless_out(P2, mode, im, f, F))
        # the identity holds in the torus T = R/Z, where the LWE phase lives:
        # compare mod 1, or a large amplification makes `got` wrap while the raw
        # functional does not (34 apart at p = 101, MVM1).
        push!(res, ("T2", "output error = Tr(U* e), $mode",
                    abs(mod1c(got - dot(functional_vector(R, U), eacc))), 1e-9))
    end

    H = draw(P.nz, 1.0, MersenneTwister(2), 4000)          # (N, 4000)
    T = R.Vs * (R.starD * H)                               # tau(Omega*_0 h)
    C = (T * T') ./ size(T, 2)
    d = real(diag(C))
    push!(res, ("N0", "key noise: tau(Omega*_0 h) white",
                norm(C - Diagonal(d)) / mean(d) / R.N, 0.05))
    push!(res, ("N0", "key noise: diagonal spread", std(d) / mean(d), 0.06))

    function pipeline_err(PP, kk, Sl, mode, masks; rms::Bool = false)
        v = Float64[]
        for (i, A) in enumerate(masks)
            mu = mods(mod(i - 1, p), p) / p
            b = mod1c(dot(A, kk.s) + mu)
            _, _, (ah, bh) = bootstrap(PP, mode, A, b, kk, Sl, f, F)
            push!(v, abs(mod1c(lwe_phase(ah, bh, kk) - mu)))
        end
        return rms ? sqrt(mean(v .^ 2)) : maximum(v)
    end
    zmasks = [zeros(Float64, P.n) for _ in 1:24]
    # T1b averages a *deterministic* rounding, whose rms over a few masks is
    # dominated by whichever mask sits nearest a rounding boundary: 24 masks give
    # a worst case of 0.48 over 18 draws, 96 masks a worst case of 0.20.
    gmasks = [mod1c(rand(rng, 0:(R.M - 1), P.n) ./ R.M) for _ in 1:96]
    for mode in MODES
        push!(res, ("T1a", "zero-mask pipeline, $mode",
                    pipeline_err(P, K, S, mode, zmasks), 1e-12))
    end
    # T1b : the gadget decomposition itself, which is where the chain's remaining
    # inexactness comes from.  Deterministic and tolerance-free: x = sum_t d_t D^-t
    # + delta with |delta| <= D^-ell/2, and digits in [-D/2, D/2].  The comparison
    # is made in the torus: at the boundary |x| ~ 1/2 the greedy expansion leaves a
    # carry of exactly 1, which is 0 in T = R/Z, where the accumulator lives.
    #
    # We do NOT test the chain-level scaling of the output error in ell.  It looks
    # like the natural test and it is a trap: the rotated polynomial has all its
    # coefficients exactly in (1/p)Z, so the FIRST external product decomposes a
    # number whose rounding residue is (D^ell mod p)/p -- deterministic, and varying
    # with ell (1/7, 4/7, 2/7, 1/7 for D = 32, p = 7, ell = 3..6).  Any two-point
    # ratio in ell therefore carries a systematic bias that no amount of averaging
    # over masks removes, and which is largest for the sparsest rotated polynomial.
    for (Dlog, ell) in ((5, 4), (7, 6))
        Pg = Params(p, alpha; sig_e = 0.0, sig_bk = 0.0, Dlog = Dlog, ell = ell)
        x = mod1c(rand(rng, 64 * R.N) .- 0.5)
        d = decomp(Pg, x)
        rec = sum(d[t] .* Float64(Pg.D)^(-t) for t in 1:Pg.ell)
        push!(res, ("T1b", "gadget remainder <= D^-l/2, D=2^$Dlog",
                    max(0.0, maximum(abs.(mod1c(x .- rec))) /
                             (Float64(Pg.D)^(-Pg.ell) / 2) - 1.0), 1e-9))
        push!(res, ("T1b", "gadget digits balanced, D=2^$Dlog",
                    max(0.0, maximum(maximum(abs.(dt)) for dt in d) - Pg.D / 2), 1e-12))
    end
    # T1c : the chain-level consequence, as a one-sided bound immune to that bias.
    Pa = Params(p, alpha; sig_e = 0.0, sig_bk = 0.0, Dlog = 5, ell = 4)
    Pb = Params(p, alpha; sig_e = 0.0, sig_bk = 0.0, Dlog = 5, ell = 6)
    Sa = bootstrap_keys(Pa, K, MersenneTwister(1))
    Sb = bootstrap_keys(Pb, K, MersenneTwister(1))
    for mode in MODES
        ea = pipeline_err(Pa, K, Sa, mode, gmasks; rms = true)
        eb = pipeline_err(Pb, K, Sb, mode, gmasks; rms = true)
        push!(res, ("T1c", "gadget error falls with l, $mode",
                    100.0 * eb / max(ea, 1e-300), 1.0))
    end

    ok = true
    for (tag, name, e, t) in res
        ok &= (e < t)
        verbose && @printf("  [%-4s] %-34s %9.2e  (tol %7.1e)  %s\n",
                           tag, name, e, t, e < t ? "PASS" : "FAIL")
    end
    return ok
end

"a dense matrix as plain CSV, for the figure scripts"
function writemat(path::AbstractString, A::AbstractMatrix)
    open(path, "w") do io
        for i in axes(A, 1)
            for j in axes(A, 2)
                j > 1 && print(io, ",")
                @printf(io, "%.10g", A[i, j])
            end
            println(io)
        end
    end
end

"""
    run_paper(; settings, n, ell, nb, nbdec, target_sd, seed, prefix)

Every number of Section 5, from one command.  For each (p, alpha) it calibrates
the key noise so that the setting DECRYPTS -- the worst of the ten
amplifications left `target_sd` standard deviations inside the decoding radius
1/(2p) -- then measures the amplifications on nb accumulator draws and confirms
the decoding on nbdec complete bootstraps per mode.  Writes two CSV files.

The ring degree is N = (p-1) p^(alpha-1) and is printed with each setting: it is
a SECURITY parameter and the caller chooses it, this function only reports it.
"""
# The five settings of Section 5.  N = (p-1) p^(alpha-1) is the SECURITY
# parameter and is chosen here, not derived: 2500, 2058, 1210, 2028, 1332.

const PAPER = ((5, 5), (7, 4), (11, 3), (13, 3), (37, 2))

# Table 1 measures the FULL N x N covariance of e*, which needs many more than N
# draws to be estimated at all; at N = 2058 that is out of reach.  The covariance
# settings are therefore small rings -- N <= 110 -- and this is a constraint of
# estimation, not a choice of convenience.
const COVPAPER = ((7, 1), (11, 1), (17, 1), (5, 2), (7, 2), (3, 4))

function run_paper(; settings = PAPER, n::Int = 512, ell::Int = 6, Dlog::Int = 7,
                   nb::Int = 3000, nbdec::Int = 200, target_sd::Real = 8.0,
                   seed::Int = 11, covsettings = COVPAPER, nbcov::Int = 6000,
                   ncov::Int = 128, ellcov::Int = 5, sigcov = 2.0^-26,
                   prefix::AbstractString = joinpath(@__DIR__, "mvm_paper"))
    amp = NamedTuple[]; dec = NamedTuple[]; cal = NamedTuple[]
    # It is the PRODUCT D^ell that sets the gadget noise floor, not ell: halving
    # ell at fixed D^ell halves the cost of every external product and the size
    # of the key spectra, at the price of digits D times larger and hence of a
    # larger floating-point rounding error in the convolutions.  (ell, Dlog) =
    # (6, 7) and (3, 14) are the same 2^42; the second is cheaper and noisier.
    @printf("threads %d   FFTW %s   n = %d, ell = %d, D = 2^%d (D^ell = 2^%d), nb = %d\n",
            Threads.nthreads(), HAVE_FFTW ? "yes" : "no", n, ell, Dlog, ell * Dlog, nb)
    for (p, a) in settings
        c = calibrate_sigbk(p, a; n = n, ell = ell, Dlog = Dlog,
                            target_sd = target_sd, seed = seed)
        N = (p - 1) * p^(a - 1)
        @printf("\n=== p = %d, alpha = %d   N = %d   M = %d   sig_bk = 2^%.0f   (worst amplification %.1f)\n",
                p, a, N, p^a, log2(c.sig_bk), c.amax)
        push!(cal, (p = p, alpha = a, N = N, n = n, ell = ell, Dlog = Dlog,
                    sig_bk = c.sig_bk, amax = c.amax))
        for md in MODELS                      # the same setting under both models
            append!(amp, exp_amplification(p, a; nb = nb, n = n, model = md,
                                           sig_bk = md === :rounding ? 0.0 : c.sig_bk,
                                           ell = ell, Dlog = Dlog, seed = seed))
        end
        append!(dec, exp_decrypt(p, a; sig_bk = c.sig_bk, n = n, ell = ell,
                                 Dlog = Dlog, nb = nbdec, seed = seed))
    end
    r = [x.measured / x.predicted for x in amp]
    @printf("\n  measured/predicted over %d rows: mean %.4f, sd %.3f%%, worst %.3f%%\n",
            length(r), mean(r), 100 * std(r), 100 * maximum(abs.(r .- 1)))
    @printf("  decryption failures: %d out of %d complete bootstraps, and %d of %d\n",
            sum(x.fails for x in dec), sum(x.nb for x in dec),
            sum(x.fails for x in amp), length(amp) * nb)
    @printf("  worst decoding margin: %.1f sd (target %.1f)\n",
            minimum(x.margin for x in amp), target_sd)
    open(prefix * "_amp.csv", "w") do io
        println(io, "p,alpha,N,n,nb,map,mode,model,sigma1,sigma2,measured,predicted,radius,margin,worst,fails")
        for x in amp
            @printf(io, "%d,%d,%d,%d,%d,%s,%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d\n",
                    x.p, x.alpha, x.N, x.n, x.nb, x.map, x.mode, x.model, x.sigma1, x.sigma2,
                    x.measured, x.predicted, x.radius, x.margin, x.worst, x.fails)
        end
    end
    open(prefix * "_dec.csv", "w") do io
        println(io, "p,alpha,mode,map,nb,sigma,radius,worst,margin,fails")
        for x in dec
            @printf(io, "%d,%d,%s,%s,%d,%.8g,%.8g,%.8g,%.8g,%d\n",
                    x.p, x.alpha, x.mode, x.map, x.nb, x.sigma, x.radius,
                    x.worst, x.margin, x.fails)
        end
    end
    open(prefix * "_cal.csv", "w") do io
        println(io, "p,alpha,N,n,ell,Dlog,sig_bk,amax")
        for x in cal
            @printf(io, "%d,%d,%d,%d,%d,%d,%.8g,%.8g\n",
                    x.p, x.alpha, x.N, x.n, x.ell, x.Dlog, x.sig_bk, x.amax)
        end
    end
    # ---------------------------------------------- the covariance, Table 1
    println("\n--- covariance of e* (Table 1), on the small rings")
    cov = [exp_covariance(p, a; nb = nbcov, n = ncov, ell = ellcov, model = md,
                          sig_bk = md === :rounding ? 0.0 : sigcov, seed = seed + 10)
           for (p, a) in covsettings for md in MODELS]
    println("\n", table1(cov))
    open(prefix * "_table1.tex", "w") do io; print(io, table1_tex(cov)); end
    open(prefix * "_cov.csv", "w") do io
        println(io, "p,alpha,N,n,nb,mode,model,diag,band,other,target,res_model,res_white,floor,emax")
        for x in cov
            @printf(io, "%d,%d,%d,%d,%d,%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
                    x.p, x.alpha, x.N, x.n, x.nb, x.mode, x.model, x.diag, x.band, x.other,
                    x.target, x.res_spherical, x.res_white, x.floor, x.emax)
        end
    end
    for x in cov                      # the matrices themselves, for the figures
        writemat(@sprintf("%s_G_%s_%d_%d.csv", prefix, x.model, x.p, x.alpha), x.G)
        writemat(@sprintf("%s_Gth_%s_%d_%d.csv", prefix, x.model, x.p, x.alpha), x.Gth)
    end
    @printf("  written: %s_{amp,dec,cal,cov}.csv, %s_table1.tex, and %d G matrices\n",
            prefix, prefix, 2 * length(cov))
    return (amp = amp, dec = dec, cal = cal, cov = cov)
end

# =================================================================== main ====

function main(args)
    what = isempty(args) ? "test" : args[1]
    quick = length(args) > 1 && args[2] == "quick"
    @printf("threads %d   FFTW %s\n", Threads.nthreads(), HAVE_FFTW ? "yes" : "no")
    Threads.nthreads() == 1 &&
        println("  (start julia with -t auto to use every core)")
    if what == "test"
        ok = true
        for (p, a) in ((7, 1), (11, 1), (101, 1), (11, 2))
            @printf("=== p = %d, alpha = %d\n", p, a)
            ok &= selftest(p, a)
        end
        println(ok ? "\nall tests PASS" : "\nsome tests FAILED")
    elseif what == "run"
        nb  = quick ? 800 : 8000
        nbc = quick ? 800 : 6000
        println("\n--- amplification (Figure 3)")
        amp = NamedTuple[]
        for (p, a) in ((7, 2), (17, 1))
            append!(amp, exp_amplification(p, a; nb = nb, n = 256, seed = 11))
        end
        r = [x.measured / x.predicted for x in amp]
        @printf("  ratio measured/predicted: mean %.4f, sd %.3f%%, worst %.3f%%\n",
                mean(r), 100 * std(r), 100 * maximum(abs.(r .- 1)))
        println("\n--- covariance (Table 1)")
        cov = [exp_covariance(p, a; nb = nbc, n = 128, seed = 21)
               for (p, a) in ((7, 1), (11, 1), (17, 1), (5, 2), (7, 2), (3, 4))]
        println("\n", table1(cov))
        open("mvm_jl_table1.tex", "w") do io; print(io, table1_tex(cov)); end
        println("\n", figures(amp, cov))
    else
        println("usage: julia mvm_modes.jl [test | run [quick]]")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
