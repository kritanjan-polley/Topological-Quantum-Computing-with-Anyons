using LinearAlgebra
using Printf
using DelimitedFiles
using ITensorMPS
using ITensors

include("functions.jl")
using .functions

start_time = time()

model = length(ARGS) >= 1 ? strip(ARGS[1]) : "model2"
@printf("Initial Peak: %.5f GB\n", get_peak_memory_bytes() / (1024^3))

const kB = 1.0
const tensor_cutoff = 1e-6
const max_bond_dim::Int64 = 128
const bath_modes::Int64 = 100

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

function run_spin_boson_mps(J_func::Function, M::Int,
    omega_max::Float64, temperature::Float64, d::Int=4)

    epsilon, Delta = 1.0, 1.0
    mu12, mu13 = 1.0, -0.2
    dt = 0.05
    t_max = model == "model2" ? 5.0 : 150.0
    sk_depth = 4
    db = generate_database(12)

    kappa, Omega, t_hop = map_continuous_bath(J_func, M, omega_max, temperature)
    sites = [Index(j == 1 ? 3 : d, "Site, n=$j") for j in 1:M+1]

    a_b, adag_b, N_b, X_b = boson_operators(d)
    I_b, I_sys = Matrix{Float64}(I, d, d), Matrix{Float64}(I, 3, 3)
    H_sys = [0.0 0.0 0.0;
            0.0 epsilon Delta;
            0.0 Delta -epsilon]
    sigma_z_SB = [0.0 0.0 0.0;
                0.0 1.0 0.0;
                0.0 0.0 -1.0]

    exact_gates_half, exact_gates_full = ITensor[], ITensor[]
    comp_gates_half, comp_gates_full = ITensor[], ITensor[]

    println("Compiling braiding gates")
    for j in 1:M
        s1, s2 = sites[j], sites[j+1]
        if j == 1
            H_bond = kron(I_b, H_sys) + kappa * kron(X_b, sigma_z_SB) + kron(Omega[1] * N_b, I_sys)
        else
            H_bond = t_hop[j-1] .* (kron(a_b, adag_b) .+ kron(adag_b, a_b)) .+ kron(Omega[j] .* N_b, I_b)
        end

        U_half_mat = exp(-im * H_bond * dt / 2.0)
        U_full_mat = exp(-im * H_bond * dt)

        U_comp_half_mat = compile_local_unitary(U_half_mat, db, sk_depth)
        U_comp_full_mat = compile_local_unitary(U_full_mat, db, sk_depth)

        push!(exact_gates_half, itensor(reshape(U_half_mat, dim(s1), dim(s2), dim(s1), dim(s2)),
            s1', s2', s1, s2))
        push!(exact_gates_full, itensor(reshape(U_full_mat, dim(s1), dim(s2), dim(s1), dim(s2)),
            s1', s2', s1, s2))
        push!(comp_gates_half, itensor(reshape(U_comp_half_mat, dim(s1), dim(s2), dim(s1), dim(s2)),
            s1', s2', s1, s2))
        push!(comp_gates_full, itensor(reshape(U_comp_full_mat, dim(s1), dim(s2), dim(s1), dim(s2)),
            s1', s2', s1, s2))
    end

    mu_matrix = [0.0 mu12 mu13;
                mu12 0.0 0.0;
                mu13 0.0 0.0]
    mu_op = itensor(mu_matrix, sites[1]', sites[1])

    psi_ex_vac = MPS(sites, fill(1, length(sites)))
    psi_ex_t   = apply(mu_op, psi_ex_vac)
    psi_ex_ref = copy(psi_ex_vac)

    psi_cp_t   = copy(psi_ex_t)
    psi_cp_ref = copy(psi_ex_vac)

    time_array = range(0.0, t_max, step=dt)
    C_t_exact, C_t_comp = ComplexF64[], ComplexF64[]

    println("Running Dual MPS Evolution...")
    for t in time_array
        push!(C_t_exact, inner(psi_ex_ref, apply(mu_op, psi_ex_t)))
        push!(C_t_comp,  inner(psi_cp_ref, apply(mu_op, psi_cp_t)))

        #  trotter MPS
        for ps in [psi_ex_t, psi_ex_ref]
            ps[:] = apply(exact_gates_half[1:2:end], ps; cutoff=tensor_cutoff, maxdim=max_bond_dim)
            ps[:] = apply(exact_gates_full[2:2:end], ps; cutoff=tensor_cutoff, maxdim=max_bond_dim)
            ps[:] = apply(exact_gates_half[1:2:end], ps; cutoff=tensor_cutoff, maxdim=max_bond_dim)
        end

        # trotter braid
        for ps in [psi_cp_t, psi_cp_ref]
            ps[:] = apply(comp_gates_half[1:2:end], ps; cutoff=tensor_cutoff, maxdim=max_bond_dim)
            ps[:] = apply(comp_gates_full[2:2:end], ps; cutoff=tensor_cutoff, maxdim=max_bond_dim)
            ps[:] = apply(comp_gates_half[1:2:end], ps; cutoff=tensor_cutoff, maxdim=max_bond_dim)
        end

        if floor(t/dt) % 5 == 0
            @printf("t=%8.3f | Exact D=%4d | Comp D=%4d | Elapsed time=%10.5f\n",
                t, maxlinkdim(psi_ex_t), maxlinkdim(psi_cp_t), time()-start_time)
	    flush(stdout)
        end
    end

    data_file = "spectra_spin_boson_$(model).txt"
    open(data_file, "w") do io
        for i in eachindex(time_array)
            @printf(io, "%.6e  %.10e  %.10e  %.10e  %.10e\n",
                time_array[i],
                real(C_t_exact[i]), imag(C_t_exact[i]),
                real(C_t_comp[i]),  imag(C_t_comp[i])
            )
        end
    end

    @printf("Memory at the end: %.5f GB\n", get_peak_memory_bytes() / (1024^3))
    @printf("It took: %.4f seconds\n", time() - start_time)
end

omega_c, lamda = 1.0, (model == "model2" ? 2.0 : 0.125)
T_env = model == "model2" ? 2.0 : 0.2
J_ohmic(w) = (2.0 * lamda * omega_c) .* w ./ (abs2.(w) .+ omega_c^2)

run_spin_boson_mps(J_ohmic, bath_modes, 20.0, T_env, 10)
