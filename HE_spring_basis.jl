using LinearAlgebra
using CairoMakie
using Printf

start_time = time()

function exact_radial_correlation()
    println("Setting up radial grid")
    N = 4000
    L = 40.0
    dr = L / (N + 1)
    r = (1:N) .* dr

    println("Building Hamiltonians")
    T = zeros(Float64, N, N)
    for i in 1:N
        T[i, i] = 2.0 / dr^2
        if i > 1
            T[i, i-1] = -1.0 / dr^2
        end
        if i < N
            T[i, i+1] = -1.0 / dr^2
        end
    end

    omega = 0.5
    V_base = (1.0 ./ r) .+ (0.25 * omega^2 .* r .^ 2)

    H0 = T + Diagonal(V_base)
    H1 = T + Diagonal(V_base .+ 2.0 ./ r .^ 2)

    println("Diagonalizing Hamiltonians")
    E0, U0 = eigen(H0)
    E1, U1 = eigen(H1)
    # E0, U0 = LAPACK.stev!('V', diag(H0), diag(H0, 1))
    # E1, U1 = LAPACK.stev!('V', diag(H1), diag(H1, 1))

    for n in 1:N
        U0[:, n] ./= sqrt(sum(abs2.(U0[:, n])) * dr)
        U1[:, n] ./= sqrt(sum(abs2.(U1[:, n])) * dr)
    end

    println("Preparing initial state")
    r_eq = (2.0 / omega^2)^(1 / 3)
    sigma = 1.0
    u_initial = r .* exp.(-((r .- r_eq) .^ 2) ./ (2 * sigma^2))
    u_initial = ComplexF64.(u_initial)
    u_initial ./= sqrt(sum(abs2.(u_initial)) * dr)

    c0 = U0' * u_initial * dr
    D = U1' * Diagonal(r) * U0 * dr
    d1 = D * c0

    println("Propagating C(t)")
    dt = 0.05
    t_total = 100.0
    time_array = range(0.0, t_total, step=dt)


    T_row = reshape(time_array, 1, :)
    Mat_A = conj.(c0) .* exp.(im .* E0 .* T_row)
    Mat_N = d1 .* exp.(-im .* E1 .* T_row)
    C_t_matrix = sum(Mat_N .* (D * Mat_A), dims=1)
    C_t = dropdims(C_t_matrix, dims=1)

    filename = "HEatom_corr_wavefunction.txt"
    open(filename, "w") do io
        for t in eachindex(time_array)
            @printf(io, "%16.8f %16.8f %16.8f\n",
                time_array[t], real(C_t[t]), imag(C_t[t]))
        end
    end

    @printf("Propagation took: %.5f seconds\n", time() - start_time)
    println("Plotting")
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"C(t)")
    lines!(ax, time_array, real.(C_t), label="Real Part")
    lines!(ax, time_array, imag.(C_t), label="Imaginary Part")
    axislegend(ax, position=:rt)

    save("HEatom_wavefunction.pdf", fig)
    @printf("It took: %.5f seconds\n", time() - start_time)

    display(fig)
end

exact_radial_correlation()
