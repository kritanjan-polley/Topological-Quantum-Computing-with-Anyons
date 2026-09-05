using LinearAlgebra
using Printf
using StaticArrays
using CairoMakie

# units
using PhysicalConstants.CODATA2022
using Unitful
using UnitfulAtomic

include("functions.jl")
using .functions

start_time = time()

pic_dir = "../pic_dir"
data_dir = "../data_dir"

set_theme!(theme_latexfonts())

const hbar::Float64 = ReducedPlanckConstant.val # 6.62607015e-34 / 2pi # J-s
const kB::Float64 = BoltzmannConstant.val # 1.380649e-23    # J/K
const mass::Float64 = ProtonMass.val # 9.1093837139e-31 * 1836.0  # 1836 au in kg
const j_to_ev::Float64 = uconvert(u"eV", 1u"J").val # 6.241509e18
const speed_of_light_cm::Float64 = SpeedOfLightInVacuum.val * 100.0 # 2.99792458e10 # cm/s
const fs_to_au::Float64 = austrip(1.0u"fs") #41.341374575751
const bohr_to_m::Float64 = BohrRadius.val # 5.29177210544e-11

wavenumbers_to_rads(wn::Float64) = 2pi * speed_of_light_cm * wn
get_characteristic_length(omega::Float64, m::Float64)::Float64 = sqrt(hbar / (m * omega))

function potential(x::Float64; m=mass, omega=1.0, lambda=0.0)
    return -0.5 * m * omega^2 * x^2 + lambda * x^4
end

function get_hamil(; nvib::Int=10, m=mass, omega=1.0, lambda=0.0)
    sqrt_nums = sqrt.(range(1, nvib - 1, step=1.0))
    a_op = diagm(1 => sqrt_nums)
    adag_op = a_op'

    x_zpf = sqrt(hbar / (2.0 * m * omega))
    X_op = x_zpf .* (a_op + adag_op)

    p_prefactor = im * sqrt((m * hbar * omega) / 2.0)
    P_op = p_prefactor .* (adag_op - a_op)
    H_kin = (P_op * P_op) ./ (2.0 * m)
    X2 = X_op * X_op
    X4 = X2 * X2

    H_pot_quadratic = -(0.5 * m * omega^2) .* X2
    H_pot_quartic = lambda .* X4

    H_total = H_kin + H_pot_quadratic + H_pot_quartic

    return H_total, a_op, adag_op, X_op
end

function get_hamil_grid(x_grid; m=mass, omega=1.0, lambda=0.0, x_displacement=0.0)
    N_grid = length(x_grid)
    dx = step(x_grid) # abs(x_grid[2] - x_grid[1])

    X_op = diagm(0 => x_grid)
    V_diag = potential.(x_grid, m=m, omega=omega, lambda=lambda)
    V_mat = diagm(0 => V_diag)

    T_mat = zeros(Float64, N_grid, N_grid)
    P_mat = zeros(ComplexF64, N_grid, N_grid)

    # colbert-miller
    prefactor_T = hbar^2 / (2.0 * m * dx^2)
    prefactor_P = -im * hbar / dx

    for i in 1:N_grid, j in 1:N_grid
        if i == j
            T_mat[i, j] = prefactor_T * (pi^2 / 3.0)
            P_mat[i, j] = 0.0
        else
            diff = i - j
            sign_diff = (-1)^diff
            T_mat[i, j] = prefactor_T * (2.0 * sign_diff) / (diff^2)
            P_mat[i, j] = prefactor_P * sign_diff / diff
        end
    end

    H_total = T_mat + V_mat
    D_op = exp(-im * P_mat * x_displacement / hbar)
    @assert ishermitian(H_total)

    return H_total, X_op, D_op
end


function single_simulation(; time_step_fs=1.0, sk_db_len=14, sk_scan_depth=6,
    tol_10=8.0, io=io)
    nvib_basis::Int = 50
    omega_cm = 500.0
    V0_cm = 1500.0
    temperature = 350.0
    dt::Float64 = time_step_fs * 1e-15
    t_max = 180.0e-15

    sk_depth::Int = sk_scan_depth # max
    sk_database_length::Int = sk_db_len
    test_tolerance::Float64 =  1.0 * 10.0^(-tol_10)

    omega_basis = wavenumbers_to_rads(omega_cm)
    V0_joules = hbar * wavenumbers_to_rads(V0_cm)
    lambda = (mass^2 * omega_basis^4) / (16.0 * V0_joules)
    beta = 1.0 / (kB * temperature)
    x_char = get_characteristic_length(omega_basis, mass)

    @printf("Mass: %.4e kg\n", mass)
    @printf("Frequency: %.2e cm inverse\n", omega_cm)
    @printf("Char. Length: %.4e a.u.\n", x_char * 1e15 * fs_to_au)
    println("Temperature: $(temperature) K")
    @printf("Barrier height = %.2e cm inverse\n", V0_cm)
    @printf("System size: %d\n", nvib_basis)
    @printf("Timestep = %.5f fs\n", dt * 1e15)

    grid_limit = 30.0 * x_char
    @printf("Grid length %.4e a.u.\n", 2.0 * grid_limit / bohr_to_m)

    N_grid = 1001
    x_grid = range(-grid_limit, grid_limit, length=N_grid)
    H_grid, X_grid_op, _ = get_hamil_grid(x_grid, m=mass,
        omega=omega_basis, lambda=lambda)

    evals_g, evecs_g = eigen(H_grid)
    @printf("The minimum eigen value is %.6e\n", minimum(evals_g))
    evals_g .-= minimum(evals_g)

    probs_g = exp.(-beta .* evals_g)
    Z_g = sum(probs_g)
    rho_energy_diag = Diagonal(probs_g ./ Z_g)
    X_energy_basis = evecs_g' * X_grid_op * evecs_g

    H_basis, _, _, X_basis_op = get_hamil(nvib=nvib_basis, m=mass,
        omega=omega_basis, lambda=lambda)
    U_step_basis = exp(im * H_basis * dt / hbar)

    println("Solovay-Kitaev iteration depth $(sk_depth)")
    @printf("Using tolerance value for SK is: %.5e\n", test_tolerance)
    db = generate_database(sk_database_length)
    gates_list, D_exact = decompose_unitary(U_step_basis)

    println("Found $(length(gates_list)) two-level rotations.")
    @printf("Memory used: %.5f GB\n", get_peak_memory_bytes() / (1024^3))

    compiled_U_step = Matrix{ComplexF64}(I, nvib_basis, nvib_basis)
    total_braiding_len::Int = 0
    for (i, j, u_target) in reverse(gates_list)
        u_approx, path = solovay_kitaev(u_target, sk_depth, db; tol=test_tolerance)
        path = simplify_path(path)
        total_braiding_len += length(path)
        G_approx = embed(u_approx, nvib_basis, i, j)
        compiled_U_step = G_approx * compiled_U_step
    end
    compiled_U_step = compiled_U_step * D_exact
    overlap = tr(U_step_basis' * compiled_U_step) / nvib_basis
    infidelity = 1.0 - abs2(overlap)
    @printf("Total braiding sequence length: %d\n", total_braiding_len)
    @printf("Braid per rotation: %.5f\n", total_braiding_len / length(gates_list))
    @printf("Braid infidelity: %.4e\n", infidelity)

    @printf(io, "%.4f %5d %.4e %10d %10.4f %6d\n",
        time_step_fs, sk_database_length, test_tolerance,
        total_braiding_len, total_braiding_len / length(gates_list),
        length(gates_list)
    )

    evals_b, evecs_b = eigen(H_basis)
    probs_b = exp.(-beta .* (evals_b .- minimum(real.(evals_b))))
    rho_b = evecs_b * Diagonal(probs_b ./ sum(probs_b)) * evecs_b'
    A_compiled = X_basis_op * rho_b

    time_array = range(0.0, t_max, step=dt)
    c_t_grid = Vector{ComplexF64}(undef, length(time_array))
    c_t_braid = similar(c_t_grid)

    println("Running time evolution")
    for (t_idx, t) in enumerate(time_array)
        U_diag = exp.(-im .* evals_g .* t ./ hbar)
        X_t = Diagonal(conj.(U_diag)) * X_energy_basis * Diagonal(U_diag)
        c_t_grid[t_idx] = tr(X_t * X_energy_basis * rho_energy_diag)
        c_t_braid[t_idx] = tr(A_compiled * X_basis_op)
        A_compiled = compiled_U_step * A_compiled * compiled_U_step'
    end

    t_au = time_array .* (1e15 * fs_to_au)
    c_t_grid = c_t_grid ./ real(c_t_grid[1])
    c_t_braid = c_t_braid ./ real(c_t_braid[1])

    prefix = @sprintf("dt_%.2f_dblen_%d_tol_-%.2f_temp_%d",
		      time_step_fs, sk_db_len, log10(test_tolerance),
        round(Int, temperature))
    data_file = joinpath(data_dir,
        "corr_doublewell_" * prefix * "_data.txt")
    open(data_file, "w") do io
        for i in eachindex(t_au)
            @printf(io, "%.6e  %.10e  %.10e  %.10e  %.10e\n",
                t_au[i], real(c_t_braid[i]), imag(c_t_braid[i]),
                real(c_t_grid[i]), imag(c_t_grid[i]))
        end
    end

    fig = Figure(size=(1200, 400))

    ax1 = Axis(fig[1, 1],
        xlabel=L"t\, (\textrm{a.u.})",
        ylabel=L"\textrm{Re}\, C_{xx}(t)",
        xtickformat=x -> [@sprintf("%.0f", val) for val in x],)
    lines!(ax1, t_au, real.(c_t_grid), label="Exact (DVR)")
    lines!(ax1, t_au, real.(c_t_braid), linestyle=:dash, label="Braiding")
    axislegend(ax1)

    ax2 = Axis(fig[1, 2],
        xlabel=L"t\, (\textrm{a.u.})",
        ylabel=L"\textrm{Im}\, C_{xx}(t)",
        xtickformat=x -> [@sprintf("%.0f", val) for val in x],)
    lines!(ax2, t_au, imag.(c_t_grid), label="Exact (DVR)")
    lines!(ax2, t_au, -imag.(c_t_braid), linestyle=:dash, label="Braiding")
    # axislegend(ax2)

    ax3 = Axis(fig[1, 3],
        xlabel=L"t\, (\textrm{a.u.})",
        ylabel=L"|C_{xx}(t)|",
        xtickformat=x -> [@sprintf("%.0f", val) for val in x],)
    lines!(ax3, t_au, abs.(c_t_grid), label="Exact (DVR)")
    lines!(ax3, t_au, abs.(c_t_braid), linestyle=:dash, label="Braiding")

    pic_file = joinpath(data_dir,
        "auto_corr_double_well_" * prefix * "_plot.pdf")
    save(pic_file, fig)
    # display(fig)
end

function main()
    if length(ARGS) != 3
        println("Usage: julia script_name.jl <time_step_fs> <sk_db_len> <tol_10>")
        println("Example: julia script_name.jl 1.0 12 10.0")
        return
    end

    time_step_fs = parse(Float64, ARGS[1])
    sk_db_len = parse(Int64, ARGS[2])
    tol_10 = parse(Float64, ARGS[3])

    start_time = time()

    id = get(ENV, "SLURM_ARRAY_TASK_ID", "0")
    track_sheet = "benchmark_braiding_doublewell_$(id).txt"
    if isfile(track_sheet)
        io = open(track_sheet, "a")
    else
        io = open(track_sheet, "w")
    end
    single_simulation(; time_step_fs=time_step_fs, sk_db_len=sk_db_len, sk_scan_depth=6,
             tol_10=tol_10, io=io)
    @printf("Done: t=%.2f fs, database len = %d, tol = 10^(%d) after %.5f seconds\n",
        time_step_fs, sk_db_len, round(Int64, tol_10), time() - start_time)
    close(io)

    @printf("It took %.4f seconds\n", time() - start_time)
end

main()
