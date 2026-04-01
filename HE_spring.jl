using FFTW
using CairoMakie
using LinearAlgebra
using Printf

function model_he_spring()
    start_time = time()

    omega = 0.5
    epsilon = 0.0
    N::Int64 = 2^7
    L = 30.0
    dx = L / N

    grid_1d = range(-L / 2 + dx / 2, step=dx, length=N)

    x = reshape(grid_1d, N, 1, 1)
    y = reshape(grid_1d, 1, N, 1)
    z = reshape(grid_1d, 1, 1, N)

    p_1d = fftfreq(N, 2pi / dx)
    px = reshape(p_1d, N, 1, 1)
    py = reshape(p_1d, 1, N, 1)
    pz = reshape(p_1d, 1, 1, N)

    T_val = abs2.(px) .+ abs2.(py) .+ abs2.(pz)
    r_sq = abs2.(x) .+ abs2.(y) .+ abs2.(z)
    s = sqrt.(r_sq .+ epsilon^2)
    V = (1.0 ./ s) .+ ((0.25 * omega^2) .* r_sq)

    dt = 0.05
    t_total = 100.0
    nstep = round(Int64, t_total / dt)

    gradV_sq = r_sq .* (-1.0 ./ s .^ 3 .+ (0.5 * omega^2)) .^ 2
    V_TI = V .- (dt^2 / 12.0) .* gradV_sq

    U_T = exp.(-(im * dt) .* T_val)
    U_V_half = exp.(-(im * dt / 2.0) .* V_TI)

    println("Initializing wave functions")
    r_eq = (2.0 / omega^2)^(1 / 3)
    sigma = 1.0
    r_mag_initial = sqrt.(r_sq)
    psi = exp.(-((r_mag_initial .- r_eq) .^ 2) ./ (2 * sigma^2)) .+ 0.0im

    norm_factor = sqrt(sum(abs2.(psi)) * dx^3)
    psi ./= norm_factor

    psi0 = copy(psi)
    phi_x = x .* psi0

    time_array = Float64[]
    C_t = ComplexF64[]

    println("Starting propagation ($nstep steps)")

    for step in 1:nstep
        # propagate |psi(0)>
        psi .*= U_V_half
        psi_p = fft(psi)
        psi_p .*= U_T
        psi = ifft(psi_p)
        psi .*= U_V_half

        # propagate x|psi(0)>
        phi_x .*= U_V_half
        phi_x_p = fft(phi_x)
        phi_x_p .*= U_T
        phi_x = ifft(phi_x_p)
        phi_x .*= U_V_half

        corr_x = sum(conj.(psi) .* x .* phi_x) * dx^3

        push!(C_t, 3.0 * corr_x)
        push!(time_array, step * dt)

        if step % 1000 == 0
            @printf("Step %d / %d completed\n", step, nstep)
        end
    end

    filename = "HEatom_corr_FFT.txt"
    open(filename, "w") do io
        for t in eachindex(time_array)
            @printf(io, "%16.8f %16.8f %16.8f\n",
                time_array[t], real(C_t[t]), imag(C_t[t]))
        end
    end

    println("Plotting")
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel=L"t", ylabel=L"C(t)")

    lines!(ax1, time_array, real.(C_t), label="Real Part")
    lines!(ax1, time_array, imag.(C_t), label="Imaginary Part")
    axislegend(ax1, position=:rt)

    @printf("It took: %.2f seconds\n", time() - start_time)
    save("HEatom_FFT.pdf", fig)
    display(fig)
end

model_he_spring()
