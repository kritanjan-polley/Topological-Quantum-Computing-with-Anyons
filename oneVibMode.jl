using LinearAlgebra
using Printf
using StaticArrays
using CairoMakie

# customs ones
include("functions.jl")
using .functions

set_theme!(merge(theme_latexfonts(), custom_theme))


@printf("Initial Peak: %.6f GB\n", get_peak_memory_bytes() / (1024^3))

function get_hamil(; e1=0.0, e2=0.5, jval=0.5, nvib::Int=10,
    omega=0.5, lambda=0.6, g=1.0)
    h_en = [e1 jval; jval e2]

    I_en = Matrix{Float64}(I, 2, 2)
    I_vib = Matrix{Float64}(I, nvib, nvib)
    sqrt_nums = sqrt.(range(1, nvib - 1, step=1.0))
    a_op = diagm(1 => sqrt_nums)
    adag_op = a_op'

    x_op = (a_op + adag_op) ./ sqrt(2.0 * omega)
    p_op = im * sqrt(omega / 2.0) .* (adag_op - a_op)

    H_kin = (p_op * p_op) .* 0.5
    H_pot_harmonic = (0.5 * omega^2) .* (x_op * x_op)
    H_pot_quartic = lambda .* (x_op^4)

    h_vib = H_kin + H_pot_harmonic + H_pot_quartic
    sigz = [1.0 0.0; 0.0 -1.0]
    H_int = (g * sqrt(2.0)) .* kron(sigz, x_op)

    H_total = kron(h_en, I_vib) + kron(I_en, h_vib) + H_int

    return H_total
end

function get_hamil_dvr(; nx::Int=512, xmax=20.0, omega=1.0,
    lambda=0.8, g=0.1, J=0.5, E1=0.0, E2=0.8)
    x = range(-xmax, xmax, length=nx)
    dx = step(x)

    T = zeros(Float64, nx, nx)
    factor = inv(2.0 * dx^2)
    for i in 1:nx, j in 1:nx
        if i == j
            T[i, j] = factor * (pi^2 / 3.0)
        else
            diff = i - j
            T[i, j] = factor * (2.0 * (-1)^diff) / (diff^2)
        end
    end

    V_base = (0.5 * omega^2) .* (x .^ 2) .+ (lambda .* (x .^ 4))
    V1 = V_base .+ E1 .+ (g .* x .* sqrt(2.0))
    V2 = V_base .+ E2 .- (g .* x .* sqrt(2.0))

    H11 = T + diagm(V1)
    H22 = T + diagm(V2)
    H12 = diagm(fill(J, nx))

    H_total = [H11 H12; H12 H22]

    return H_total
end



function run_dvr_dynamics(t_list::AbstractArray; nx::Int=512, xmax=20.0,
    omega=1.0, lambda=0.8, g=0.1, J=0.5, E1=0.0, E2=0.8)

    H = get_hamil_dvr(nx=nx, xmax=xmax, omega=omega, lambda=lambda,
        g=g, J=J, E1=E1, E2=E2)
    x_grid = range(-xmax, xmax, length=nx)
    psi_initial_spatial = (omega / pi)^(0.25) .* exp.(-0.5 * omega .* x_grid .^ 2)

    dx = step(x_grid)
    norm_factor = sqrt(sum(abs2, psi_initial_spatial) * dx)
    psi_initial_spatial = psi_initial_spatial ./ norm_factor

    psi_0 = [psi_initial_spatial; zeros(ComplexF64, nx)]
    @assert ishermitian(H)
    evals, evecs = eigen(H)
    c0 = evecs' * psi_0

    pop_1 = zeros(length(t_list))
    pop_2 = similar(pop_1)

    for (i, t) in enumerate(t_list)
        c_t = c0 .* exp.(-im .* evals .* t)
        psi_t = evecs * c_t
        psi_1 = psi_t[1:nx]
        psi_2 = psi_t[nx+1:end]
        pop_1[i] = sum(abs2, psi_1) * dx
        pop_2[i] = sum(abs2, psi_2) * dx
    end

    return pop_1, pop_2
end

function plot_nuclear_potentials(ax; xmax=6.0, nx=501,
                                   omega=1.0, lambda=0.25, g=0.3,
                                   e1=0.4, e2=-0.4, jval=0.5)

    x = range(-xmax, xmax, length=nx)
    V_base = @. (0.5 * omega^2 * x^2) + (lambda * x^4)
    V1 = @. V_base + e1 + (g * x * sqrt(2.0))
    V2 = @. V_base + e2 - (g * x * sqrt(2.0))
    diff = V1 .- V2
    disc = @. sqrt(diff^2 + 4.0 * jval^2)

    E_lower = @. 0.5 * (V1 + V2 - disc)
    E_upper = @. 0.5 * (V1 + V2 + disc)

    # diabatics
    lines!(ax, x, V1, linestyle=:dash, label=L"V_{11} (x)")
    lines!(ax, x, V2, linestyle=:dash, label=L"V_{22} (x)")
    # Plot Adiabatics (Solid)
    lines!(ax, x, E_lower, label="Adiabatic Lower")
    lines!(ax, x, E_upper, label="Adiabatic Upper")

    axislegend(ax, position=:ct, nbanks=2,
        orientation = :horizontal, patchsize = (40, 20))

    return ax
end

function main()
    start_time = time()
    e1 = 0.4
    e2 = -0.4
    jval = 0.5
    omega = 1.0
    lambda = 0.25
    g = 0.3
    nvib = 16
    N_states = 2
    N = N_states * nvib
    println("System Size: $N (Electronic: $N_states, Vibrational: $nvib)")

    db = generate_database(14)
    sk_recursion = 3
    H = get_hamil(nvib=nvib, e1=e1, e2=e2, jval=jval,
        omega=omega, lambda=lambda, g=g)

    dt = 0.05
    time_array = range(0, 20.0, step=dt)
    U_step_exact = exp(im * H * dt)

    println("Decomposing unitary into 2-level gates")
    gates_list, D_exact = decompose_unitary(U_step_exact)
    println("Got $(length(gates_list)) two-level rotations.")

    println("Getting sub-gates with Solovay-Kitaev")
    compiled_U_step = Matrix{ComplexF64}(I, N, N)
    total_braids::Int = 0
    printed_sample::Bool= false

    for (idx, (i, j, u_target)) in enumerate(reverse(gates_list))
        u_approx, path = solovay_kitaev(u_target, sk_recursion, db)
        path = simplify_path(path)
        total_braids += length(path)

        if !printed_sample
            if total_braids > 50
                println("\nBraiding sequence (first processed gate):")
                println(path[1:50])
                println("... (remaining elements omitted)")
                printed_sample = true
            end
        end

        G_approx = embed(u_approx, N, i, j)
        compiled_U_step = G_approx * compiled_U_step
    end
    compiled_U_step = compiled_U_step * D_exact

    @printf("Total no of braids: %d\n", total_braids)

    rho_exact = zeros(ComplexF64, N, N)
    rho_exact[1, 1] = 1.0
    rho_compiled = rho_exact

    pop_exact = zeros(length(time_array), 2)
    pop_compiled = similar(pop_exact)

    pop_dvr = similar(pop_exact)
    pop_dvr[:, 1], pop_dvr[:, 2] = run_dvr_dynamics(time_array;
        nx=501, xmax=30.0, omega=omega, lambda=lambda, g=g,
        J=jval, E1=e1, E2=e2)

    for (t_idx, t) in enumerate(time_array)
        diag_exact = real.(diag(rho_exact))
        pop_exact[t_idx, 1] = sum(diag_exact[1:nvib])
        pop_exact[t_idx, 2] = sum(diag_exact[nvib+1:end])

        diag_compiled = real.(diag(rho_compiled))
        pop_compiled[t_idx, 1] = sum(diag_compiled[1:nvib])
        pop_compiled[t_idx, 2] = sum(diag_compiled[nvib+1:end])

        rho_exact = U_step_exact' * rho_exact * U_step_exact
        rho_compiled = compiled_U_step' * rho_compiled * compiled_U_step
    end

    unitary_error = opnorm(U_step_exact - compiled_U_step)
    overlap = tr(U_step_exact' * compiled_U_step) / N
    infidelity = 1.0 - abs2(overlap)
    @printf("Unitary Approximation Error (Spectral Norm):\n")
    @printf("|| U_exact - U_approx || = %.4e\n", unitary_error)
    @printf("Infidelity: %.4e\n", infidelity)

    println("Plotting results")
    begin
        println("Plotting diabatic and adiabatic potential")
        fig = Figure(linewidth=2)
        ax = Axis(fig[1,1], xlabel=L"x", ylabel=L"V(x)")
        plot_nuclear_potentials(ax; nx=501, xmax=2.0, omega=omega,
            lambda=lambda, g=g, e1=e1, e2=e2, jval=jval)
        save("2LS_potential.pdf", fig)
        fig
    end

    fig = Figure()
    ax11 = Axis(fig[1, 1], xlabel=L"t/\hbar", ylabel="Population")

    # lines!(ax11, time_array, pop_exact[:, 1], color=:blue,
    #     label=L"\rho_{11}\,\textrm{(Exact)}")
    lines!(ax11, time_array, pop_dvr[:, 1], color=my_colors[1],
        label=L"\rho_{11}\,\textrm{(Exact)}")
    scatter!(ax11, time_array, pop_compiled[:, 1], color=my_colors[1],
        label=L"\rho_{11}\,\textrm{(Braid)}")

    # lines!(ax11, time_array, pop_exact[:, 2], color=:red,
    #     label=L"\rho_{22}\,\textrm{(Exact)}")
    lines!(ax11, time_array, pop_dvr[:, 2], color=my_colors[2],
        label=L"\rho_{22}\,\textrm{(Exact)}")
    scatter!(ax11, time_array, pop_compiled[:, 2], color=my_colors[2],
        label=L"\rho_{22}\,\textrm{(Braid)}")

    axislegend(ax11, framevisible=false, nbanks=2)

    save("testVib.pdf", fig)
    @printf("Memory at the end: %.6f GB\n", get_peak_memory_bytes() / (1024^3))
    @printf("\nIt took: %.4f seconds\n", time() - start_time)
    return fig
end

main()
