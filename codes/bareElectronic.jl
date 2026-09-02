using LinearAlgebra
using Printf
using CairoMakie
using LaTeXStrings

include("functions.jl")
using .functions

pic_dir = "../pic_dir"
#
# include("functionsIsing.jl")
# using .functionsIsing

set_theme!(merge(theme_latexfonts(), custom_theme))

@printf("Initial Peak: %.6f GB\n", get_peak_memory_bytes() / (1024^3))

function main()
    start_time = time()
    N::Int = 2
    @assert N >= 2
    println("System Size: $N")

    sk_recursion::Int = 8 # max depth
    database_length::Int = 14
    db::ForwardDB = generate_database(database_length)

    if N == 2
        H = [0.5 0.75;
            0.75 -0.5]
        print_len = 10
    elseif N == 3
        H = [0.0 0.9 0.7;
            0.9 0.5 0.6;
            0.7 0.6 0.8]
        print_len = 30
    elseif N >= 50
        H_rand = sprand(N, N, 0.09234)
        H_rand = Array(H_rand)
        H = H_rand + H_rand'
        H = H ./ norm(H) .* Float64(N)
        print_len = 30
    else
        H_rand = rand(Float64, N, N)
        H = H_rand + H_rand'
        H = H ./ norm(H) .* Float64(N)
        print_len = 30
    end
    @assert ishermitian(H)
    if N <= 15
        println("The Hamiltonian is")
        display(H)
    end

    dt = 0.05
    time_array = range(0, 10.0, step=dt)

    U_step_exact::Array{ComplexF64,2} = exp(im * H * dt)
    gates_list, D_exact = decompose_unitary(U_step_exact)
    println("Found $(length(gates_list)) two-level rotations.")
    println("Getting sub-gates with Solovay-Kitaev")

    compiled_U_step = Matrix{ComplexF64}(I, N, N)
    total_braids::Int = 0
    printed_sample::Bool = false
    full_braiding_sequence = Symbol[]

    for (idx, (i, j, u_target)) in enumerate(reverse(gates_list))
        u_approx, path::Vector{Symbol} = solovay_kitaev(u_target, sk_recursion, db, tol=1e-7)
        path = simplify_path(path)
        total_braids += length(path)

        if !printed_sample
            if total_braids > print_len
                cheeck_len = min(print_len, total_braids)
                println("\nBraiding sequence (first processed gate):")
                println(path[1:cheeck_len])
                println("... (remaining elements omitted)")
                printed_sample = true
            end
        end

        append!(full_braiding_sequence, path)

        G_approx = embed(u_approx, N, i, j)
        compiled_U_step = G_approx * compiled_U_step
    end
    compiled_U_step = compiled_U_step * D_exact
    if length(full_braiding_sequence) <= 400
        println("\n" * "-"^80)
        println("Full Braiding Sequence (LaTeX format):")
        println(to_latex_string(full_braiding_sequence))
        println("-"^80 * "\n")
    else
        println("\n" * "-"^80)
        println(to_latex_string(full_braiding_sequence[1:100]))
        println("\n" * "-"^80)
    end
    @printf("Total no of braids: %d\n", total_braids)

    unitary_error = opnorm(U_step_exact - compiled_U_step)
    overlap = tr(U_step_exact' * compiled_U_step) / N
    infidelity = 1.0 - abs2(overlap)
    @printf("Unitary Approximation Error (Spectral Norm):\n")
    @printf("|| U_exact - U_approx || = %.4e\n", unitary_error)
    @printf("Infidelity: %.4e\n", infidelity)

    rho_exact = zeros(ComplexF64, N, N)
    rho_exact[1, 1] = 1.0
    rho_compiled = rho_exact

    pop_exact = zeros(length(time_array), N)
    pop_compiled = zeros(length(time_array), N)

    for (t_idx, t) in enumerate(time_array)
        pop_exact[t_idx, :] = real.(diag(rho_exact))
        pop_compiled[t_idx, :] = real.(diag(rho_compiled))
        rho_exact = U_step_exact' * rho_exact * U_step_exact
        rho_compiled = compiled_U_step' * rho_compiled * compiled_U_step
    end

    println("Plotting results")
    fig = Figure()
    ax11 = Axis(fig[1, 1], xlabel=L"t/\hbar", ylabel="Population")

    plot_limit = min(N, 4)

    for k in 1:plot_limit
        c = my_colors[k]
        label_exact = latexstring("\\textrm{Exact} (\\rho_{$k$k})")
        label_braid = latexstring("\\textrm{Braid} (\\rho_{$k$k})")
        lines!(ax11, time_array, pop_exact[:, k], color=c, label=label_exact)
        scatter!(ax11, time_array, pop_compiled[:, k], color=c, label=label_braid)
    end

    if N <= 2
        axislegend(ax11, nbanks=2, framevisible=false)
    else
        Legend(fig[1, 2], ax11, nbanks=1, tellheight=false)
        colsize!(fig.layout, 1, Relative(0.75))
    end
    save(joinpath(pic_dir, "testBare$(N)LS.pdf"), fig)
    @printf("Memory at the end: %.6f GB\n", get_peak_memory_bytes() / (1024^3))
    @printf("It took %.4f seconds\n", time() - start_time)
    return fig
end

main()
