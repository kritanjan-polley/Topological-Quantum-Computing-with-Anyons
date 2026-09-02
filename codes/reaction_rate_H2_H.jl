using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using CUDA

include("LSTH.jl")
using .LSTH

include("functions.jl")
using .functions

# BLAS.set_num_threads(4)
start_time = time()

@printf("Memory usage = %.3f GB\n", Sys.maxrss()/(1024^3))

const hbar = 1.0
const kB = 3.166811563e-6
const mH = 1837.152647
const mur = mH/2
const muR = 2mH/3
const au2_cms = 2.18769126364e8

function make_pes(r, R; rmin=0.2, Vwall=5.0)
    build_lsth()
    V = zeros(length(r), length(R))

    for j in eachindex(R), i in eachindex(r)
        d1 = r[i]
        d2 = abs(R[j] - r[i]/2)
        d3 = abs(R[j] + r[i]/2)

        V[i,j] = min(d1, d2, d3) < rmin ? Vwall : lsth(d1, d2, d3)
    end
    return V
end

function kinetic(x, mu)
    n, dx = length(x), x[2] - x[1]
    f, T = 1/(2.0*mu * dx^2), zeros(n, n)

    for i in eachindex(x)
        T[i,i] = f * pi^2 / 3
        for j in i+1:n
            d = i - j
            T[i,j] = T[j,i] = 2.0*f * (-1.0)^d / d^2
        end
    end
    return T
end

function hamiltonian(r, R, V)
    nr, nR = length(r), length(R)
    Tr, TR = kinetic(r, mur), kinetic(R, muR)

    H = Matrix(
        kron(sparse(I, nR, nR), sparse(Tr)) +
        kron(sparse(TR), sparse(I, nr, nr))
    )

    H[diagind(H)] .+= vec(V)
    return H
end

function reactant_data(r, V, T)
    H_react = kinetic(r, mur) + Diagonal(V[:, end])
    evals = eigvals(Symmetric(H_react))
    E0 = minimum(evals)
    beta = 1.0 / (kB * T)

    qvib = sum(exp.(-beta .* (evals .- E0)))
    qtrans = sqrt(muR / (2pi * beta * hbar^2))
    return qtrans * qvib, E0
end

function adiabatic_barrier(r, V)
    Tr = kinetic(r, mur)
    e0 = [minimum(eigvals(Tr + Diagonal(V[:, j]))) for j in axes(V, 2)]
    return maximum(e0) - (e0[1] + e0[end]) / 2
end

function product_projector_vec(r, R)
    nr = length(r)
    h = zeros(Float64, nr * length(R))

    for j in eachindex(R), i in eachindex(r)
        k = i + (j - 1) * nr

        dAB = r[i]
        dBC = abs(R[j] - r[i]/2)
        dAC = abs(R[j] + r[i]/2)

        h[k] = !(dAB <= dBC && dAB <= dAC) ? 1.0 : 0.0
    end
    return h
end

function cff_unitary(H_mat, h_vec, T, E0; dt=0.05, tmax=3000.0)
    beta = 1.0 / (kB * T)
    H_shifted = H_mat - E0 * I
    B = exp(-0.25 * beta * H_shifted)
    H_h = H_mat .* h_vec'
    h_H = h_vec .* H_mat
    F = (im / hbar) .* (H_h .- h_H)

    A0 = B * F * B
    U_dt = exp(-im * H_mat * (dt / hbar))
    U_dt_dag = U_dt'

    t = range(0.0, tmax, step=dt)
    C = zeros(Float64, length(t))

    A_t = copy(A0)
    for n in eachindex(t)
        C[n] = real(tr(A0 * A_t))
        A_t = U_dt_dag * A_t * U_dt
    end

    return t, C
end


function cff_unitary(H, h, T, E0; dt=0.05, tmax=3000.0)
    Hg, hg = CuArray(H), CuArray(h)
    E, V = CUDA.eigen(Symmetric(Hg))

    F = V' * ((im/hbar) .* Hg .* (hg' .- hg)) * V
    b = exp.(-(E .- E0) ./ (4kB*T))
    A = F .* b .* b'
    W = A .* transpose(A)

    p = exp.(-im*dt/hbar .* (E .- E0))
    q = CUDA.ones(ComplexF64, length(E))
    qc, tmp = similar(q), similar(q)
    t = 0.0:dt:tmax
    C = zeros(Float64, length(t))

    println("Propagating $(length(t)) steps...")
    t0 = time()
    pint = max(1, div(length(t), 100))

    for n in eachindex(t)
        qc .= conj.(q)
        mul!(tmp, W, qc)
        C[n] = real(sum(q .* tmp))
        q .*= p

        if n == 1 || n % pint == 0 || n == length(t)
            @printf("Step %8d/%8d | t = %8.2f au | %5.1f%% | %8.2f sconds\n",
                    n, length(t), t[n], 100n/length(t), time()-t0)
        end
    end

    return t, C
end

function integrate_cff(t, C)
    out = zeros(length(t))
    for i in 2:length(t)
        out[i] = out[i-1] + 0.5 * (C[i-1] + C[i]) * (t[i] - t[i-1])
    end
    return out
end

function first_plateau(t, C, I_arr; frac=0.05, ctol=2e-3, itol=5e-3)
    n = length(t)
    w = max(20, round(Int, frac * n))
    i0 = max(2, round(Int, 0.05 * n))

    Cs = max(maximum(abs.(C)), eps())
    Is = max(maximum(abs.(I_arr)), eps())

    for i in i0:n-w
        j = i + w - 1
        if mean(abs.(C[i:j])) / Cs < ctol && abs(I_arr[j] - I_arr[i]) / Is < itol
            return mean(I_arr[i:j]), std(I_arr[i:j]), i, j, true
        end
    end

    score(i) = mean(abs.(C[i:i+w-1])) / Cs + abs(I_arr[i+w-1] - I_arr[i]) / Is
    candidates = i0:n-w
    i = candidates[argmin(score.(candidates))]
    j = i + w - 1

    return mean(I_arr[i:j]), std(I_arr[i:j]), i, j, false
end

function compute_rate(r, R, V; T=300.0, dt=0.05, tmax=3000.0)
    Qr, E0 = reactant_data(r, V, T)
    barrier = adiabatic_barrier(r, V)

    @printf("Temperature = %.2f K\n", T)
    println("Constructing Hamiltonian and Unitary Operator")

    H = hamiltonian(r, R, V)
    h_vec = product_projector_vec(r, R)

    println("Propagating via Unitary Matrix Multiplication")
    t, C = cff_unitary(H, h_vec, T, E0; dt=dt, tmax=tmax)
    out = integrate_cff(t, C)

    P, dP, i, j, ok = first_plateau(t, C, out)

    k = (P / Qr) / 2.0
    kcms = k * au2_cms

    @printf("E0 = %.10f Eh\n", E0)
    @printf("Adiabatic barrier = %.8f Eh = %.4f kcal/mol\n", barrier, 627.509474 * barrier)
    @printf("Q_r = %.12e\n", Qr)
    @printf("Plateau = %.12e +/- %.3e\n", P, dP)
    @printf("Plateau time = %.1f -- %.1f au\n", t[i], t[j])
    @printf("k(T) = %.12e a.u.\n", k)
    @printf("k(T) = %.12e cm/s\n", kcms)

    open(@sprintf("lsth_rate_T_%d.txt", round(Int, T)), "w") do io
        println(io, "# t   Cff   integral")
        for n in eachindex(t)
            @printf(io, "%.8e %.14e %.14e\n", t[n], C[n], out[n])
        end
    end

    return t, C, out, k
end


temps = sort(vcat(150.0:30.0:450.0, 500.0:150.0:1250.0, 1500.0))
T = isempty(ARGS) ? 200.0 : temps[parse(Int, ARGS[1])]

r = range(0.5, 12.0, length=101)
R = range(-10.0, 10.0, length=121)

println("Building lsth PES on $(length(r)) x $(length(R)) grid...")
V = make_pes(r, R)

@printf("PES minimum = %.10f Eh\n", minimum(V))
@printf("PES maximum = %.10f Eh\n", maximum(V))

t, Cff, kint, k = compute_rate(r, R, V; T=T, dt=0.05, tmax=3000.0)

@printf("Memory usage = %.3f GB\n", Sys.maxrss()/(1024^3))
@printf("Runtime = %.2f s\n", time() - start_time)
