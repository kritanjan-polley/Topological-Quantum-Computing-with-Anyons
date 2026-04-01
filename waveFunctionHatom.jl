using CairoMakie
using SpecialFunctions
using SphericalHarmonics
using GSL
using Statistics
using LinearAlgebra
using Printf

m_e = 1.0
m_p = 1836.15
a0 = 1.0

function reduced_electron_nucleus_mass(Z::Int,
    M::Union{AbstractFloat,Nothing}=nothing)
    if isnothing(M)
        if Z == 1
            M = m_p
        else
            throw(ArgumentError("'M' must be provided if Z > 1"))
        end
    end
    return (m_e * M) / (m_e + M)
end

reduced_bohr_radius(mu::Float64) = a0 * m_e / mu

function radial_wavefunction_Rnl(
    n::Int,
    l::Int,
    r::AbstractArray{<:Real};
    Z::Int=1,
    use_reduced_mass::Bool=false,
    M::Union{Real,Nothing}=nothing
)
    if !(n >= 1 && 0 <= l <= n - 1)
        throw(DomainError((n, l),
            "Quantum numbers must satisfy n ≥ 1 and 0 ≤ l ≤ n-1"))
    end

    mu = use_reduced_mass ? reduced_electron_nucleus_mass(Z, M) : m_e
    a_mu = reduced_bohr_radius(mu)

    rho = @. 2.0 * Z * r / (n * a_mu)

    L_poly = map(x -> sf_laguerre_n(n - l - 1, 2 * l + 1, x), rho)

    log_pref = 1.5 * log(2.0 * Z / (n * a_mu))
    log_pref += 0.5 * (loggamma(n - l) - (log(2.0 * n) + loggamma(n + l + 1)))
    pref = exp(log_pref)

    R = @. pref * exp(-rho / 2.0) * (rho^l) * L_poly
    return R
end

function spherical_harmonic_Ylm(l::Int, m::Int, theta::AbstractArray{<:Real},
    phi::AbstractArray{<:Real})
    if !(l >= 0 && -l <= m <= l)
        throw(DomainError((l, m),
            "Quantum numbers must satisfy l >= 0 and -l <= m <= l"))
    end
    return @. SphericalHarmonics.sphericalharmonic(theta, phi, l, m)
end

function simulate_hydrogen_basis_au(n_max::Int)
    start_time = time()

    N::Int64 = 2^7
    L = 30.0
    grid_1d = range(-L / 2, L / 2, length=N)
    dx = grid_1d[2] - grid_1d[1]

    x = reshape(grid_1d, N, 1, 1)
    y = reshape(grid_1d, 1, N, 1)
    z = reshape(grid_1d, 1, 1, N)

    r = @. sqrt(x^2 + y^2 + z^2)
    theta = @. acos(clamp(z / (r + (r == 0)), -1.0, 1.0))
    phi = @. mod2pi(atan(y, x))

    x0, y0, z0 = 0.0, 0.0, 0.0
    r = @. sqrt(abs2(x - x0) + abs2(y - y0) + abs2(z - z0))
    psi_0 = @. exp(-1.8 * r) + 0.5 * exp(-0.8 * r) + 0.4 * exp(-r^2 / 4) + 0.0im

    norm_factor = sqrt(dot(psi_0, psi_0) * dx^3)
    psi_0 ./= norm_factor

    coefficients = ComplexF64[]
    energies = Float64[]
    basis_functions = Vector{Array{ComplexF64,3}}()
    bound_state_norm = 0.0

    println("Projecting initial wavepacket onto exact states up to n = $n_max")
    for n in 1:n_max
        E_n = -1.0 / (2.0 * n * n)

        for l in 0:(n-1)
            for m in -l:l
                R_part = radial_wavefunction_Rnl(n, l, r; Z=1, use_reduced_mass=false)
                Y_part = spherical_harmonic_Ylm(l, m, theta, phi)
                psi_nlm = R_part .* Y_part
                c_nlm = dot(psi_nlm, psi_0) * dx^3

                if abs(c_nlm) > 1e-9
                    push!(coefficients, c_nlm)
                    push!(energies, E_n)
                    push!(basis_functions, psi_nlm)

                    bound_state_norm += abs2(c_nlm)
                    @printf("State |%d, %d, % d) : coeff^2 = %16.10f\n",
                        n, l, m, abs2(c_nlm))
                end
            end
        end
    end

    @printf("\nTotal norm captured by bound states up to n=%d: %.4f\n",
        n_max, bound_state_norm)

    t_target = 2.0
    println("Evolving analytically to t = $t_target a.u.")

    psi_t = zeros(ComplexF64, size(psi_0))
    for i in eachindex(coefficients)
        c_t = coefficients[i] * exp(-im * energies[i] * t_target)
        psi_t .+= c_t .* basis_functions[i]
    end

    println("Plotting density")
    z_index = round(Int64, N * 0.5)
    prob_density_slice = abs2.(psi_t[:, :, z_index])

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        title="Probability Density at t = $t_target",
        xlabel=L"\,\,\textrm{(a.u.)}", ylabel=L"y\,\,\textrm{(a.u.)}",
        aspect=DataAspect())

    hm = image!(ax, grid_1d[1] .. grid_1d[end], grid_1d[1] .. grid_1d[end],
        prob_density_slice, colormap=:jet1)
    Colorbar(fig[1, 2], hm)

    @printf("It took: %.5f seconds\n", time() - start_time)
    save("slice_Hatom_basis.pdf", fig)
    display(fig)
    # return grid_1d, psi_t
end

simulate_hydrogen_basis_au(8)
