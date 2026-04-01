using LinearAlgebra
using Printf
using DelimitedFiles
using ITensorMPS
using ITensors
using CairoMakie

include("functions.jl")
using .functions

start_time = time()

set_theme!(merge(theme_latexfonts(), custom_theme))
@printf("Initial Peak: %.6f GB\n", get_peak_memory_bytes() / (1024^3))

const kB = 1.0
const tensor_cutoff = 1e-8
const max_bond_dim::Int64 = 128

ITensors.op(::OpName"Z", ::SiteType"Site") = pauliZ

function star_to_chain(omegas, couplings)
    N = length(omegas)
    kappa = norm(couplings)
    v_n = couplings ./ kappa

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
    temperature::Float64; N_grid::Int=10001)
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


function run_spin_boson_mps(J_func::Function, M::Int, omega_max::Float64,
    temperature::Float64, d::Int=4)
    epsilon = 1.0
    J_tunn = 1.0
    dt = 0.02
    t_max = 5.0
    sk_depth = 4
    db = generate_database(12)

    println("Generating Orthogonal Polynomials for T=$(temperature) bath (M=$M modes)")
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
            H_sys = epsilon * pauliZ + J_tunn * pauliX
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
    @printf("Elapsed time (braiding): %.4f seconds\n", time() - start_time)

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

    println("Running MPS and braiding Evolution")
    for t in time_array
        # println(t)
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
    end

    data_file = "check_spin_boson_model2.txt"
    open(data_file, "w") do io
        for i in eachindex(time_array)
            @printf(io, "%.6e  %.10e  %.10e\n",
                time_array[i], real(sz_t_exact[i]), real(sz_t_comp[i])
            )
        end
    end

    heom_data = readdlm("heom_spin_boson_model2.txt")

    println("Plotting figure")
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"\langle \sigma_z(t) \rangle")

    lines!(ax, time_array, sz_t_exact, label="Exact TEBD", linewidth=2, linestyle=:dash)
    lines!(ax, time_array, sz_t_comp, label="Compiled TEBD")
    lines!(ax, heom_data[:, 1], heom_data[:, 2] .- heom_data[:, 3], label="HEOM", linewidth=1.5, linestyle=:dashdot)

    axislegend(ax)
    # display(fig)
    fig

    @printf("Memory at the end: %.6f GB\n", get_peak_memory_bytes() / (1024^3))
    @printf("It took: %.4f seconds\n", time() - start_time)
end


omega_c = 1.0
lamda = 1.0 / 8.0
J_ohmic(w) = (2.0 * lamda * omega_c) .* w ./ (abs2.(w) .+ omega_c^2)

M_modes = 100
omega_maximum = 20.0 * omega_c
T_env = omega_c / 5.0
boson_dim = 10

run_spin_boson_mps(J_ohmic, M_modes, omega_maximum, T_env, boson_dim)
