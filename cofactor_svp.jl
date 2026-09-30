#!/usr/bin/env julia
#
#   Stage (a) of Section 5: the cofactor at a random lift, in the spherical model.
#
#   requires:  import Pkg; Pkg.add("Hecke")
#
#   usage:   julia cofactor_svp.jl            sweeps every odd prime p <= 97, alpha = 1
#            julia cofactor_svp.jl p [alpha]  one setting, e.g. julia cofactor_svp.jl 13 2
#
#   Conventions of the paper.
#     M = p^alpha,  m = M/p,  N = (p-1)m;  the index splits as i = k*m + j with
#     0 <= k <= p-2 and 0 <= j < m;  gamma = sum_{i<N} g_i Omega_i, Omega_i = X^i.
#
#   Lemma 3.  Gr((Omega_0^*)^{-1} Omega) = M G (x) I_m, with G := Psi Delta^{-1} Psi
#   integral of size p-1,
#     G_{k,k'} = p m^2 ( (p^2-1)/12 - c(p-c)/2 ),   c = (k-k') mod p.
#
#   Corollary 1.  In units of sigma_F^2 sigma^2(E_1),
#     cost(gamma) = (2p / (M N)) * sum_{j<m} c^(j)' G c^(j),  c^(j) = (g_{k m + j})_k ,
#   and the minimum over R \ {0} is reached on a single non-zero slice: stage (a) is
#   the rank-(p-1) shortest-vector problem for G alone, whatever alpha.  The two
#   designs cost p(p+1)/6 for gamma = 1 and 2p for the pivot gamma = M Omega_0^*.

using Hecke

"Gram matrix G of Lemma 3: integral, positive definite, (p-1) x (p-1)."
function gram_G(p::Integer, alpha::Integer)
    @assert p > 2 && all(p % d != 0 for d in 2:isqrt(p)) "p must be an odd prime"
    m = big(p)^(alpha - 1)
    A = Matrix{BigInt}(undef, p - 1, p - 1)
    for k in 0:p-2, kp in 0:p-2
        c = mod(k - kp, p)
        num = p * m^2 * (big(p)^2 - 1 - 6 * c * (p - c))      # = 12 p m^2 g(c)
        @assert num % 12 == 0
        A[k+1, kp+1] = div(num, 12)
    end
    return A
end

"Cost of the cofactor gamma given by its N coordinates on (Omega_i),
 in units of sigma_F^2 sigma^2(E_1).  Exact rational."
function cost(p::Integer, alpha::Integer, g::AbstractVector{<:Integer})
    m = p^(alpha - 1); M = p * m; N = (p - 1) * m
    length(g) == N || error("gamma needs N = $N coordinates, got $(length(g))")
    A = gram_G(p, alpha)
    s = big(0)
    for j in 0:m-1
        c = BigInt[g[k*m + j + 1] for k in 0:p-2]
        s += (c' * A * c)
    end
    return (2 * big(p) * s) // (big(M) * big(N))
end

"Coordinates of gamma = 1."
function gamma_one(p::Integer, alpha::Integer)
    N = (p - 1) * p^(alpha - 1)
    g = zeros(BigInt, N); g[1] = 1
    return g
end

"Coordinates of the pivot gamma = M Omega_0^* = 1 - X^N.
 Phi_M = 0 reads sum_{k<p} X^{k m} = 0, hence X^N = - sum_{k<=p-2} X^{k m}."
function gamma_pivot(p::Integer, alpha::Integer)
    m = p^(alpha - 1); N = (p - 1) * m
    g = zeros(BigInt, N); g[1] = 2
    for k in 1:p-2
        g[k*m + 1] = 1
    end
    return g
end

"Exact solution of stage (a): the minimum of cost over R \\ {0}, and the minimal
 vectors of G (one per pair +/- v).  Each c is read back as gamma = sum_k c_k X^{k m}."
function stage_a(p::Integer, alpha::Integer = 1)
    m = p^(alpha - 1); M = p * m; N = (p - 1) * m
    L  = integer_lattice(gram = matrix(ZZ, gram_G(p, alpha)))
    mu = minimum(L)                        # min_{c != 0} c' G c, exact
    vs = shortest_vectors(L)
    num = BigInt(numerator(mu)); den = BigInt(denominator(mu))
    return (2 * big(p) * num) // (big(M) * big(N) * den), vs
end

const PRIMES97 = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                  47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)

"Sweep: solve stage (a) at every odd prime p <= 97 and compare with the two designs."
function sweep(primes = PRIMES97; alpha::Integer = 1)
    println(rpad("p", 5), rpad("minimum", 12), rpad("min of the two", 16),
            rpad("status", 12), rpad("vectors", 9), "time")
    allok = true
    for p in primes
        t = @elapsed ((mn, vs) = stage_a(p, alpha))
        pred = min((p * (p + 1)) // 6, 2p // 1)
        ok = (mn == pred); allok &= ok
        println(rpad(p, 5), rpad(string(mn), 12), rpad(string(pred), 16),
                rpad(ok ? "ok" : "DIFFERENT", 12),
                rpad(string(length(vs)), 9), round(t, digits = 2), " s")
    end
    println(allok ? "\nno cofactor beats the better of the two designs" :
                    "\nsome p give a strictly smaller minimum")
    return allok
end

function main()
    if isempty(ARGS)
        sweep()
    else
        p     = parse(Int, ARGS[1])
        alpha = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
        m = p^(alpha - 1); M = p * m; N = (p - 1) * m
        println("p = $p, alpha = $alpha, M = $M, m = $m, N = $N\n")
        println("  gamma = 1   : ", cost(p, alpha, gamma_one(p, alpha)),
                "\t predicted p(p+1)/6 = ", (p * (p + 1)) // 6)
        println("  pivot       : ", cost(p, alpha, gamma_pivot(p, alpha)),
                "\t predicted 2p        = ", 2p // 1)
        mn, vs = stage_a(p, alpha)
        println("  SVP minimum : ", mn, "\t min of the two     = ",
                min((p * (p + 1)) // 6, 2p // 1))
        println("  reached by ", length(vs), " vectors, one per pair +/- v")
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
