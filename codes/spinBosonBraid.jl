using LinearAlgebra
using Printf
using ITensorMPS
using ITensors

include("functions.jl")
using .functions

start_time = time()
data_dir = "../data_dir"

@printf("Initial Peak: %.6f GB\n", get_peak_memory_bytes() / (1024^3))

const kB = 1.0
const tensor_cutoff = 1e-7
const max_bond_dim::Int64 = 128

const pauliX::Matrix{ComplexF64} = [0.0 1.0; 1.0 0.0]
const pauliZ::Matrix{ComplexF64} = [1.0 0.0; 0.0 -1.0]
const i2::Matrix{ComplexF64} = [1.0 0.0; 0.0 1.0]

ITensors.op(::OpName"Z", ::SiteType"Site") = pauliZ

function star_to_chain(omegas, couplings)
    N = length(omegas)
    kappa = norm(couplings)
    v_n = couplings ./ kappa

    if length(omegas) != length(couplings)
        throw(ArgumentError(
            "Frequency and coupling arrays must have same length!"
            ))
    end

    Omega = zeros(Float64, N)
    t = zeros(Float64, N - 1)
    W = Diagonal(omegas)
    v_prev = zeros(Float64, N)

    for n in 1:N
        w = W * v_n
        Omega[n] = dot(v_n, w)
        w = w .- Omega[n] .* v_n
        if n > 1
            w = w .- t[n-1] .* v_prev
        end

        if n < N
            t[n] = norm(w)
            v_prev = copy(v_n)
            v_n = w ./ t[n]
        end
    end
    return kappa, Omega, t
end

function map_continuous_bath(J_func::Function, M_chain::Int, omega_max::Float64,
    temperature::Float64; N_grid::Int=1001)
    d_omega = 2.0 * omega_max / N_grid
    omegas_dense = range(-omega_max + d_omega / 2, omega_max - d_omega / 2,
        length=N_grid)
    beta = 1.0 / (kB * temperature)

    function J_thermal(w)
        if w > 1e-10
            return J_func(w) / (1.0 - exp(-beta * w))
        elseif w < -1e-10
            return J_func(abs(w)) / (exp(beta * abs(w)) - 1.0)
        else
            return J_func(1e-8) / (beta * 1e-8)
        end
    end

    couplings_dense = sqrt.(J_thermal.(omegas_dense) .* d_omega ./ pi)
    kappa, Omega_full, t_full = star_to_chain(omegas_dense, couplings_dense)
    return kappa, Omega_full[1:M_chain], t_full[1:M_chain-1]
end

function boson_operators(d::Int)
    a = zeros(Float64, d, d)
    for n in 1:(d-1)
        a[n, n+1] = sqrt(n)
    end
    I_d = Matrix{Float64}(I, d, d)
    adag = a'
    N_op = adag * a .+ 0.5 * I_d
    X_op = adag + a
    return a, adag, N_op, X_op
end

function compile_local_unitary(U_exact::Matrix{ComplexF64}, db::ForwardDB,
    sk_depth::Int)
    N = size(U_exact, 1)
    gates_list, D_exact = decompose_unitary(U_exact)
    compiled_U = Matrix{ComplexF64}(I, N, N)

    for (i, j, u_target) in reverse(gates_list)
        u_approx, path = solovay_kitaev(u_target, sk_depth, db)
        G_approx = embed(u_approx, N, i, j)
        compiled_U = G_approx * compiled_U
    end
    return compiled_U * D_exact
end

function run_spin_boson_mps(J_func::Function, M::Int, omega_max::Float64,
    temperature::Float64, d::Int=4)
    epsilon = 1.0
    jval= 2.0
    dt = 0.05
    t_max = 8.0

    sk_depth = 10
    db = generate_database(14)

    println("Generating Orthogonal Polynomials T=$(temperature) bath (M=$M modes)")
    kappa, Omega, t_hop = map_continuous_bath(J_func, M, omega_max, temperature)

    sites = [Index(j == 1 ? 2 : d, "Site, n=$j") for j in 1:M+1]

    a_b, adag_b, N_b, X_b = boson_operators(d)
    I_b = Matrix{Float64}(I, d, d)

    exact_half_gates = ITensor[]
    exact_full_gates = ITensor[]
    comp_half_gates = ITensor[]
    comp_full_gates = ITensor[]

    println("Compiling Solovay-Kitaev Gates")
    for j in 1:M
        s1 = sites[j]
        s2 = sites[j+1]

        d1 = dim(s1)
        d2 = dim(s2)

        if j == 1
            H_sys = epsilon * pauliZ + jval* pauliX
            H_bond = kron(I_b, H_sys)
            H_bond += kappa * kron(X_b, pauliZ)
            H_bond += kron(Omega[1] * N_b, i2)
        else
            H_hop = t_hop[j-1] * (kron(a_b, adag_b) + kron(adag_b, a_b))
            H_bond = H_hop + kron(Omega[j] * N_b, I_b)
        end

        U_half_exact_mat = exp(-im * H_bond * (dt / 2.0))
        U_full_exact_mat = exp(-im * H_bond * dt)

        U_half_comp_mat = compile_local_unitary(U_half_exact_mat, db, sk_depth)
        U_full_comp_mat = compile_local_unitary(U_full_exact_mat, db, sk_depth)

        push!(exact_half_gates, itensor(reshape(U_half_exact_mat, d1, d2, d1, d2), s1', s2', s1, s2))
        push!(exact_full_gates, itensor(reshape(U_full_exact_mat, d1, d2, d1, d2), s1', s2', s1, s2))

        push!(comp_half_gates, itensor(reshape(U_half_comp_mat, d1, d2, d1, d2), s1', s2', s1, s2))
        push!(comp_full_gates, itensor(reshape(U_full_comp_mat, d1, d2, d1, d2), s1', s2', s1, s2))
    end

    exact_half_odd = exact_half_gates[1:2:end]
    exact_full_even = exact_full_gates[2:2:end]

    comp_half_odd = comp_half_gates[1:2:end]
    comp_full_even = comp_full_gates[2:2:end]

    state_strings = [1 for _ in 1:M+1]
    psi_exact = MPS(sites, state_strings)
    psi_comp = MPS(sites, state_strings)

    time_array = range(0.0, t_max, step=dt)
    sz_t_exact = Float64[]
    sz_t_comp = Float64[]

    println("Starting time evolution")

    data_file = joinpath(data_dir, "check_spin_boson_model.txt")
    io = open(data_file, "w")
    for (idx, t) in enumerate(time_array)
        orthogonalize!(psi_exact, 1)
        orthogonalize!(psi_comp, 1)

        push!(sz_t_exact, expect(psi_exact, "Z"; sites=1:1)[1])
        push!(sz_t_comp, expect(psi_comp, "Z"; sites=1:1)[1])

        #  odd (dt/2) * even (dt) * odd (dt/2)
        psi_exact = apply(exact_half_odd, psi_exact; cutoff=tensor_cutoff, maxdim=max_bond_dim)
        psi_exact = apply(exact_full_even, psi_exact; cutoff=tensor_cutoff, maxdim=max_bond_dim)
        psi_exact = apply(exact_half_odd, psi_exact; cutoff=tensor_cutoff, maxdim=max_bond_dim)

        # odd (dt/2) * even (dt) * odd (dt/2)
        psi_comp = apply(comp_half_odd, psi_comp; cutoff=tensor_cutoff, maxdim=max_bond_dim)
        psi_comp = apply(comp_full_even, psi_comp; cutoff=tensor_cutoff, maxdim=max_bond_dim)
        psi_comp = apply(comp_half_odd, psi_comp; cutoff=tensor_cutoff, maxdim=max_bond_dim)

        if mod(idx-1, 50) == 0
            @printf("Time t = %10.4f after %10.4f seconds\n", t, time()-start_time)
        end

        @printf(io, "%.6e  %.10e  %.10e\n",
            t, real(sz_t_exact[idx]), real(sz_t_comp[idx])
        )
    end
    close(io)

    @printf("Memory at the end: %.6f GB\n", get_peak_memory_bytes() / (1024^3))
    @printf("It took: %.4f seconds\n", time() - start_time)
end


if abspath(PROGRAM_FILE) == @__FILE__
    omega_c = 1.0
    lamda = 1.0
    J_ohmic(w) = (2.0 * lamda * omega_c) .* w ./ (abs2.(w) .+ omega_c^2)

    M_modes = 100
    omega_maximum = 15.0 * omega_c
    T_env = omega_c * 2.0
    boson_dim = 10

    run_spin_boson_mps(J_ohmic, M_modes, omega_maximum, T_env, boson_dim)
end
