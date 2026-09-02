using LinearAlgebra
using Printf
using DelimitedFiles
using FFTW
using CairoMakie

include("functions.jl")
using .functions

BLAS.set_num_threads(2)
set_theme!(theme_latexfonts())

const if_plot = true
const start_time = time()

data_dir = "data_dir"
pic_dir = "pic_dir"

const n_orb = 8
const au_to_ev = 27.211386245988


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

occ(state, p) = (state >> p) & 1
fermion_sign(state, p) = (-1)^count_ones(state & ((1 << p) - 1))

# N=2
const valid_states = sort!([
    (1 << a) | (1 << b)
    for a in 0:(n_orb - 1) for b in (a + 1):(n_orb - 1)
])

const dim_elec = length(valid_states)
const state_to_sub = Dict(valid_states[i] => i for i in eachindex(valid_states))
const state_occs = [
    [p for p in 1:n_orb if occ(valid_states[idx], p - 1) == 1]
    for idx in 1:dim_elec
]

function build_cdag_c_op(p_1based::Int, q_1based::Int)
    p, q = p_1based - 1, q_1based - 1
    mat = zeros(ComplexF64, dim_elec, dim_elec)

    for (i, state) in enumerate(valid_states)
        if occ(state, q) == 1
            s1 = state & ~(1 << q)
            sign1 = fermion_sign(state, q)
            if occ(s1, p) == 0
                s2 = s1 | (1 << p)
                sign2 = fermion_sign(s1, p)
                if haskey(state_to_sub, s2)
                    mat[state_to_sub[s2], i] = sign1 * sign2
                end
            end
        end
    end
    return mat
end

function build_h4_op(p_1based::Int, q_1based::Int, r_1based::Int, s_1based::Int)
    p, q, r, s = (p_1based - 1, q_1based - 1, r_1based - 1, s_1based - 1)

    mat = zeros(ComplexF64, dim_elec, dim_elec)

    for (i, state) in enumerate(valid_states)
        if occ(state, s) == 1
            s1 = state & ~(1 << s)
            sgn1 = fermion_sign(state, s)
            if occ(s1, r) == 1
                s2 = s1 & ~(1 << r)
                sgn2 = fermion_sign(s1, r)
                if occ(s2, q) == 0
                    s3 = s2 | (1 << q)
                    sgn3 = fermion_sign(s2, q)
                    if occ(s3, p) == 0
                        s4 = s3 | (1 << p)
                        sgn4 = fermion_sign(s3, p)
                        if haskey(state_to_sub, s4)
                            mat[state_to_sub[s4], i] += sgn1 * sgn2 * sgn3 * sgn4
                        end
                    end
                end
            end
        end
    end
    return mat
end

const cdag_c_ops = [build_cdag_c_op(p, q) for p in 1:n_orb, q in 1:n_orb]
const nops = [cdag_c_ops[p, p] for p in 1:n_orb]
const nops_ops = [nops[p] * nops[q] for p in 1:n_orb, q in 1:n_orb]
const h4_ops = [
    build_h4_op(p, q, r, s)
    for p in 1:n_orb, q in 1:n_orb, r in 1:n_orb, s in 1:n_orb
]

const Ttot = 20_000.0
const dt = 0.025
const Nslices = Int(round(Ttot / dt))

const n_grid_vib = 128
const r_min = 0.5
const r_max = 10.0
const dr = (r_max - r_min) / (n_grid_vib - 1)
const r_grid = range(r_min, r_max, length=n_grid_vib)

const mu_h = 1836.152673 / 2.0
const dim_tot = dim_elec * n_grid_vib

function get_fft(array)
    time_array = array[:, 1]
    data_hs = array[:, 2] .+ 1im .* array[:, 4]
    data_exact = array[:, 3] .+ 1im .* array[:, 5]

    target_len = 2^22
    if length(time_array) < target_len
        excess = target_len - length(time_array)
        data_hs = [data_hs; zeros(eltype(data_hs), excess)]
        data_exact = [data_exact; zeros(eltype(data_exact), excess)]
    end

    big_len = length(data_hs)
    delta_t = abs(time_array[2] - time_array[1])
    freq = sort(fftfreq(big_len) .* (2pi / delta_t))

    fft_exact = real.(fftshift(ifft(data_exact))) .* sqrt(big_len)
    fft_hs = real.(fftshift(ifft(data_hs))) .* sqrt(big_len)

    return freq, fft_exact, fft_hs
end

const h2_table = readdlm(joinpath(data_dir, "h2_matrix_elements.txt"), comments=true)
const U_table = readdlm(joinpath(data_dir, "U_matrix_elements.txt"), comments=true)
const h4_table = readdlm(joinpath(data_dir, "h4_matrix_elements.txt"), comments=true)
const mux_table = readdlm(joinpath(data_dir, "mux_matrix_elements.txt"), comments=true)
const muy_table = readdlm(joinpath(data_dir, "muy_matrix_elements.txt"), comments=true)
const Enuc_table = readdlm(joinpath(data_dir, "Enuc_matrix_elements.txt"), comments=true)

function preprocess_table(table, n_inds)
    dict = Dict{Tuple, Tuple{Vector{Float64}, Vector{Float64}}}()

    for row in eachrow(table)
        inds = Tuple(Int.(row[1:n_inds]))
        x = Float64(row[n_inds + 1])
        y = Float64(row[n_inds + 2])

        if !haskey(dict, inds)
            dict[inds] = (Float64[], Float64[])
        end
        push!(dict[inds][1], x)
        push!(dict[inds][2], y)
    end

    for k in keys(dict)
        xs, ys = dict[k]
        p = sortperm(xs)
        dict[k] = (xs[p], ys[p])
    end
    return dict
end

function preprocess_Enuc_table(table)
    xs = Float64[]
    ys = Float64[]

    for row in eachrow(table)
        push!(xs, Float64(row[1]))
        push!(ys, Float64(row[2]))
    end

    p = sortperm(xs)
    return Dict((1,) => (xs[p], ys[p]))
end

const h2_dict = preprocess_table(h2_table, 2)
const U_dict = preprocess_table(U_table, 2)
const h4_dict = preprocess_table(h4_table, 4)
const mux_dict = preprocess_table(mux_table, 2)
const muy_dict = preprocess_table(muy_table, 2)
const Enuc_dict = preprocess_Enuc_table(Enuc_table)

function interp_from_table(dict, inds::Tuple, R::Float64)
    if !haskey(dict, inds)
        return 0.0
    end

    xs, ys = dict[inds]
    if R <= xs[1]
        return ys[1]
    elseif R >= xs[end]
        return ys[end]
    end

    k = searchsortedlast(xs, R)
    x1, x2 = xs[k], xs[k + 1]
    y1, y2 = ys[k], ys[k + 1]
    return y1 + (y2 - y1) * (R - x1) / (x2 - x1)
end

function get_ab_initio_integrals(R::Float64)
    h2 = zeros(ComplexF64, n_orb, n_orb)
    U = zeros(ComplexF64, n_orb, n_orb)
    h4 = zeros(ComplexF64, n_orb, n_orb, n_orb, n_orb)

    for i in 1:n_orb, j in 1:n_orb
        h2[i, j] = interp_from_table(h2_dict, (i, j), R)
        U[i, j] = interp_from_table(U_dict, (i, j), R)
    end

    for i in 1:n_orb, j in 1:n_orb, k in 1:n_orb, l in 1:n_orb
        h4[i, j, k, l] = interp_from_table(h4_dict, (i, j, k, l), R)
    end

    return h2, U, h4
end

function build_H4(h4)
    H4 = zeros(ComplexF64, dim_elec, dim_elec)

    for p in 1:n_orb, q in 1:n_orb, r in 1:n_orb, s in 1:n_orb
        if (p == s && q == r) || (p == r && q == s)
            continue
        end

        val = h4[p, q, r, s]
        if abs(val) > 1e-12
            H4 .+= 0.5 * val .* h4_ops[p, q, r, s]
        end
    end

    return 0.5 .* (H4 + H4')
end

function build_Helec_at_R(R::Float64)
    Helec = zeros(ComplexF64, dim_elec, dim_elec)
    h2, Umat, h4 = get_ab_initio_integrals(R)

    for p in 1:n_orb, q in 1:n_orb
        if abs(h2[p, q]) > 1e-12
            Helec .+= h2[p, q] .* cdag_c_ops[p, q]
        end
    end

    for p in 1:n_orb, q in 1:(p - 1)
        if abs(Umat[p, q]) > 1e-12
            Helec .+= Umat[p, q] .* nops_ops[p, q]
        end
    end

    Helec .+= build_H4(h4)
    return 0.5 .* (Helec + Helec')
end

E_nuc(R::Float64) = interp_from_table(Enuc_dict, (1,), R)

function build_T_nuc()
    T_nuc = zeros(ComplexF64, n_grid_vib, n_grid_vib)
    coeff = 1.0 / (2.0 * mu_h * dr^2)

    for i in 1:n_grid_vib, j in 1:n_grid_vib
        if i == j
            T_nuc[i, j] = coeff * (pi^2 / 3.0)
        else
            diff = i - j
            T_nuc[i, j] = coeff * (2.0 * (-1.0)^diff / diff^2)
        end
    end

    return T_nuc
end

function build_total_Hamiltonian()
    H_tot = zeros(ComplexF64, dim_tot, dim_tot)
    I_elec = Matrix{ComplexF64}(I, dim_elec, dim_elec)

    H_tot .+= kron(build_T_nuc(), I_elec)

    for i in 1:n_grid_vib
        R = r_grid[i]
        idx = (i - 1) * dim_elec + 1 : i * dim_elec
        H_tot[idx, idx] .+= build_Helec_at_R(R)
        H_tot[idx, idx] .+= E_nuc(R) .* I_elec
    end

    return 0.5 .* (H_tot + H_tot')
end

function onebody_propagator(u)
    U = zeros(ComplexF64, dim_elec, dim_elec)

    for ket in 1:dim_elec
        occ_ket = state_occs[ket]
        for bra in 1:dim_elec
            occ_bra = state_occs[bra]
            U[bra, ket] = det(u[occ_bra, occ_ket])
        end
    end

    return U
end

function hs_params(alpha)
    lam = acosh(1.0 / sqrt(2.0 - exp(alpha)))
    cc = -log(cosh(lam))
    return cc, lam
end

function Uint_HS_slice(Umat)
    pairs = Tuple{Int, Int, Float64}[]

    for p in 1:n_orb, q in 1:(p - 1)
        g = real(Umat[p, q])
        if abs(g) > 1e-12
            push!(pairs, (p, q, g))
        end
    end

    hs_factors = Vector{Tuple{Int, Int, ComplexF64, ComplexF64}}(
        undef,
        length(pairs),
    )

    for (a, (p, q, g)) in enumerate(pairs)
        alpha = -1im * dt * g
        cc, lam = hs_params(alpha)
        hs_factors[a] = (p, q, cc, lam)
    end

    Uint_diag = zeros(ComplexF64, dim_elec)

    for k in 1:dim_elec
        value = 1.0 + 0.0im
        state_k = valid_states[k]

        for (p, q, cc, lam) in hs_factors
            n_sum = occ(state_k, p - 1) + occ(state_k, q - 1)
            term_plus = exp((cc + lam) * n_sum)
            term_minus = exp((cc - lam) * n_sum)
            value *= 0.5 * (term_plus + term_minus)
        end

        Uint_diag[k] = value
    end

    return Diagonal(Uint_diag)
end

function build_elec_HS_step_at_R(R::Float64)
    h2, Umat, h4 = get_ab_initio_integrals(R)

    u_h2_half = exp(-1im .* h2 .* (dt / 2.0))
    U_h2_half = onebody_propagator(u_h2_half)
    U_density_HS = Uint_HS_slice(Umat)

    H4 = build_H4(h4)
    U4_half = exp(-1im .* H4 .* (dt / 2.0))
    U_nuc_phase = exp(-1im * E_nuc(R) * dt)

    return U_nuc_phase .* (
        U_h2_half * U4_half * U_density_HS * U4_half * U_h2_half
    )
end

function build_total_HS_evolution()
    U_kin_nuc_half = exp(-1im .* build_T_nuc() .* (dt / 2.0))
    I_elec = Matrix{ComplexF64}(I, dim_elec, dim_elec)
    U_kin_tot_half = kron(U_kin_nuc_half, I_elec)

    U_elec_tot = zeros(ComplexF64, dim_tot, dim_tot)
    for i in 1:n_grid_vib
        R = r_grid[i]
        idx = (i - 1) * dim_elec + 1 : i * dim_elec
        U_elec_tot[idx, idx] .= build_elec_HS_step_at_R(R)
    end

    return U_kin_tot_half * U_elec_tot * U_kin_tot_half
end

function get_dipole_integrals(dict, R::Float64)
    mu = zeros(ComplexF64, n_orb, n_orb)
    for i in 1:n_orb, j in 1:n_orb
        mu[i, j] = interp_from_table(dict, (i, j), R)
    end
    return mu
end

function build_Mu_tot(dict)
    Mu_tot = zeros(ComplexF64, dim_tot, dim_tot)

    for i in 1:n_grid_vib
        R = r_grid[i]
        mu = get_dipole_integrals(dict, R)

        Mu_elec = zeros(ComplexF64, dim_elec, dim_elec)
        for p in 1:n_orb, q in 1:n_orb
            if abs(mu[p, q]) > 1e-12
                Mu_elec .+= mu[p, q] .* cdag_c_ops[p, q]
            end
        end

        idx = (i - 1) * dim_elec + 1 : i * dim_elec
        Mu_tot[idx, idx] .= 0.5 .* (Mu_elec + Mu_elec')
    end

    return Mu_tot
end

# Dedicated test function for C_xy(t) + C_yx(t) = 0
function test_cross_terms(Mu_x_tot, Mu_y_tot, psi0, evals, vectors; n_steps=1000)
    println("="^50)
    println("checking cross terms: C_xy(t) + C_yx(t)")
    println("="^50)

    c_bra = vectors' * psi0
    c_ket_x = vectors' * (Mu_x_tot * psi0)
    c_ket_y = vectors' * (Mu_y_tot * psi0)

    max_sum_abs = 0.0

    for m in 0:n_steps
        t = m * dt
        phase = exp.((-1im * t) .* evals)

        bra_t = vectors * (phase .* c_bra)
        ket_x_t = vectors * (phase .* c_ket_x)
        ket_y_t = vectors * (phase .* c_ket_y)
        C_xy = dot(bra_t, Mu_x_tot * ket_y_t)
        C_yx = dot(bra_t, Mu_y_tot * ket_x_t)

        cross_sum = C_xy + C_yx
        max_sum_abs = max(max_sum_abs, abs(cross_sum))

        @printf("t = %6.2f | C_xy = % .4e%+.4ei | C_yx = % .4e%+.4ei | C_xy+C_yx = % .4e\n",
                t, real(C_xy), imag(C_xy), real(C_yx), imag(C_yx), abs(cross_sum))
    end

    @printf("Maximum |C_xy(t) + C_yx(t)| over test steps: %.6e\n", max_sum_abs)
    if max_sum_abs < 1e-10
        println("C_xy(t) + C_yx(t) approx 0. Cross terms vanish due to symmetry.")
    else
        println("C_xy(t) + C_yx(t) /= 0. Adding mu_x and mu_y introduces non-zero cross terms!")
    end
end

function main()
    @printf("Initial memory usage = %.5f GB\n", Sys.maxrss() / 1024.0^3)
    println("Building vibronic Hamiltonian")
    H_tot = build_total_Hamiltonian()
    @assert ishermitian(H_tot)
    println("System size = $(size(H_tot))")

    println("Diagonalizing exact vibronic Hamiltonian")
    F = eigen(Hermitian(H_tot))
    evals = real.(F.values)
    gs_idx = argmin(evals)
    psi0 = F.vectors[:, gs_idx]

    @printf("Ground-state energy = %.10f Ha\n", evals[gs_idx])
    @printf("dt = %.5f, Ttot = %.5f, Nslices = %d\n", dt, Ttot, Nslices)

    println("Building HS step and total dipole operator (Mu_x + Mu_y)")
    Ustep_HS_tot = build_total_HS_evolution()
    Mu_x_tot = build_Mu_tot(mux_dict)
    Mu_y_tot = build_Mu_tot(muy_dict)

    # mu = mu_x + mu_y
    Mu_tot = Mu_x_tot + Mu_y_tot
    test_cross_terms(Mu_x_tot, Mu_y_tot, psi0, evals, F.vectors)

    @printf("norm(Mu_tot*psi0) = %.12e\n", norm(Mu_tot * psi0))

    amps_xy = F.vectors' * (Mu_tot * psi0)
    println("Exact combined stick spectrum")
    open("stick_spectra_sigma_pi.txt", "w") do io
        for n in eachindex(evals)
            omega = evals[n] - evals[gs_idx]
            energy_ev = omega * au_to_ev
            intensity = abs2(amps_xy[n])

            if omega > 1e-8 && intensity > 1e-12
                @printf("%5d  %14.8f eV  %14.8e\n", n, energy_ev, intensity)
                @printf(io, "%14.8f  %14.8e\n", energy_ev, intensity)
            end
        end
    end

    ket0_xy = Mu_tot * psi0
    bra_HS_t = copy(psi0)
    ket_xy_HS_t = copy(ket0_xy)

    next_bra = similar(bra_HS_t)
    next_xy = similar(ket_xy_HS_t)
    tmp_xy = similar(psi0)

    c_bra_exact = F.vectors' * psi0
    c_ket_xy_exact = F.vectors' * ket0_xy

    sk_depth = 10
    db = generate_database(14)
    bra_HS_braid = compile_local_unitary(bra_HS_t, db=db, sk_depth=sk_depth)

    @printf("Starting propagation loop after %.4f seconds\n", time()-start_time)

    open("vibronic_correlation_sigma_pi.txt", "w") do io
        for m in 0:Nslices
            t = m * dt

            mul!(tmp_xy, Mu_tot, ket_xy_HS_t)
            C_HS = dot(bra_HS_braid, tmp_xy)

            phase = exp.((-1im * t) .* evals)
            bra_exact_t = F.vectors * (phase .* c_bra_exact)
            ket_xy_exact_t = F.vectors * (phase .* c_ket_xy_exact)

            mul!(tmp_xy, Mu_tot, ket_xy_exact_t)
            C_exact = dot(bra_exact_t, tmp_xy)

            if mod(m, 100) == 0
                @printf("%10.3f  % .8f  % .8f  % .8f  % .8f  %.3e  %.4f\n",
                    t, real(C_HS), real(C_exact), imag(C_HS), imag(C_exact),
                    abs(C_HS - C_exact), time()-start_time,
                )
                flush(stdout)
            end

            @printf(io, "%10.3f  % .12e  % .12e  % .12e  % .12e\n",
                t, real(C_HS), real(C_exact), imag(C_HS), imag(C_exact),
            )

            mul!(next_bra, Ustep_HS_tot, bra_HS_t)
            mul!(next_xy, Ustep_HS_tot, ket_xy_HS_t)

            bra_HS_t, next_bra = next_bra, bra_HS_t
            ket_xy_HS_t, next_xy = next_xy, ket_xy_HS_t
        end
    end

    corr = readdlm("vibronic_correlation_sigma_pi.txt")
    w, res_exact, res_hs = get_fft(corr)

    open("vibronic_corr_fft_sigma_pi.txt", "w") do io
        for i in eachindex(w)
            @printf(io, "% .12e  % .12e  % .12e\n",
                w[i], real(res_exact[i]), real(res_hs[i]),
            )
        end
    end

    if if_plot
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel="t (a.u.)", ylabel="C(t)")
        lines!(ax, corr[:, 1], corr[:, 2], label="Re (Braid)")
        lines!(ax, corr[:, 1], corr[:, 3], label="Re (exact)")
        axislegend(ax, framevisible=false)
        save("check_time_domain_sigma_pi.pdf", fig)

        fft_data = readdlm("vibronic_corr_fft_sigma_pi.txt")
        sticks = readdlm("stick_spectra_sigma_pi.txt")

        fig = Figure()
        ax = Axis(
            fig[1, 1],
            ylabel=L"I (\omega)",
            xlabel=L"\omega\;(\mathrm{eV})",
            xlabelsize=18, ylabelsize=18,
        )

        lines!(ax, fft_data[:, 1] .* au_to_ev, fft_data[:, 2], label=L"I(\omega)\;(\mathrm{Exact})")
        lines!(ax, fft_data[:, 1] .* au_to_ev, fft_data[:, 3],
            label=L"I (\omega)\;(\mathrm{Braid})", linestyle=:dashdot,
        )

        if size(sticks, 1) > 0
            max_fft = maximum(abs, fft_data[:, 2])
            max_stick = maximum(sticks[:, 2])
            scale = max_stick > 0 ? 0.8 * max_fft / max_stick : 1.0

            segments = [
                Point2f(x, y)
                for i in axes(sticks, 1)
                for (x, y) in (
                    (sticks[i, 1], 0.0),
                    (sticks[i, 1], scale * sticks[i, 2]),
                )
            ]
            linesegments!(ax, segments, label="Exact sticks")
        end

        xlims!(ax, 12.0, 15.0)
        axislegend(ax, framevisible=false)
        save("check_spectra_sigma_pi.pdf", fig)
    end

    @printf("Final memory usage = %.5f GB\n", Sys.maxrss() / 1024.0^3)
    @printf("It took %.5f seconds\n", time() - start_time)
end

main()
