using CairoMakie
using FileIO
using DelimitedFiles
using LinearAlgebra
using FFTW

base_dir = joinpath(dirname(pwd()), "data_dir");

function get_fft(array)
    hbar = 1.0
    time_array = array[:, 1]
    big_len = length(time_array)
    data_approx = array[:, 2] .+ 1im .* array[:, 3]
    data_exact = array[:, 4] .+ 1im .* array[:, 5]

    test_len = 2^20
    if big_len < test_len
        excess = test_len - big_len
        data_approx = [data_approx; zeros(eltype(data_approx), excess)]
        data_exact = [data_exact; zeros(eltype(data_exact), excess)]
    end

    big_len = length(data_approx)

    delta_t = abs(time_array[1] - time_array[2])
    scale2 = hbar * 2pi / delta_t
    freq = sort(fftfreq(big_len) .* scale2)

    fft_exct = real.(fftshift(ifft(data_exact))) .* sqrt(big_len)
    fft_approx = real.(fftshift(ifft(data_approx))) .* sqrt(big_len)

    return freq, fft_exct, fft_approx
end

lw = 1.75
ms = 10
fontsize = 24
ticksize = 20
legendsize = 20
labelsize = 22
legendlabelsize = 22
titlesize = 20

my_colors = cgrad(:glasbey_bw_minc_20_maxl_70_n256, 256, categorical=true)
custom_theme::Attributes = Theme(
    Axis=(
        xgridvisible=true,
        ygridvisible=true,
        xgridstyle=:dash,
        ygridstyle=:dash,
        xticklabelsize=ticksize,
        yticklabelsize=ticksize,
        xlabelsize=labelsize,
        ylabelsize=labelsize,
        markersize=ms,
    ),
    palette=(color=my_colors,),
    Legend=(
        titlesize=titlesize,
        labelsize=legendlabelsize,
        markersize=ms,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
    ),
)

set_theme!(merge(custom_theme, theme_latexfonts()))


# 2LS
f1 = readdlm(joinpath(base_dir, "testBare2LS_fib.txt"))
f2 = readdlm(joinpath(base_dir, "testBare2LS_ising.txt"))
img2 = load(joinpath(pwd(), "plot-1.png"))
img3 = load(joinpath(pwd(), "plottwo-1.png"))


fig = Figure(size=(1550, 350))

ax1 = Axis(fig[1, 1], xlabel=L"t/\hbar", ylabel="Population")
ax2 = Axis(fig[1, 2], aspect = DataAspect())

lines!(ax1, f1[:,1], f1[:,2], color=my_colors[1], label=L"\rho_{11} (\mathrm{Exact})")
lines!(ax1, f1[:,1], f1[:,3], color=my_colors[2], label=L"\rho_{22} (\mathrm{Exact})")

scatter!(ax1, f1[:,1], f1[:,4], color=my_colors[1], label=L"\rho_{11} (\mathrm{Braid})")
scatter!(ax1, f1[:,1], f1[:,5], color=my_colors[2], label=L"\rho_{22} (\mathrm{Braid})")

text!(ax1, 0.1, 0.05; space=:relative, text=L"\epsilon=0.5,\, J=0.6",
    align=(:left, :center), fontsize=22,
)

axislegend(ax1, "Fibonacci Anyons", nbanks=2, labelsize=18,
    framevisible=true, backgroundcolor=("white", 0.55),
    framewidth=0.5, framecolor=:gray,
    )

image!(ax2, rotr90(img2))

hidedecorations!(ax2)
hidespines!(ax2)


ax3 = Axis(fig[1, 3], xlabel=L"t/\hbar")
ax4 = Axis(fig[1, 4], aspect = DataAspect())

lines!(ax3, f2[:,1], f2[:,2], color=my_colors[1], label=L"\rho_{11} (\mathrm{Exact})")
lines!(ax3, f2[:,1], f2[:,3], color=my_colors[2], label=L"\rho_{22} (\mathrm{Exact})")

scatter!(ax3, f2[:,1], f2[:,4], color=my_colors[1], label=L"\rho_{11} (\mathrm{Braid})")
scatter!(ax3, f2[:,1], f2[:,5], color=my_colors[2], label=L"\rho_{22} (\mathrm{Braid})")

text!(ax3, 0.1, 0.05; space=:relative, text=L"\epsilon=0.4,\, J=0.6",
    align=(:left, :center), fontsize=22,
)


axislegend(ax3, "Ising Anyons", nbanks=2, labelsize=18,
    framevisible=true, backgroundcolor=("white", 0.55),
    framewidth=0.5, framecolor=:gray)

image!(ax4, rotr90(img3))

hidedecorations!(ax4)
hidespines!(ax4)


text!(ax1, 0.02, 0.9, text="(A)", space=:relative, font=:bold, fontsize=20)
text!(ax2, 0.02, 0.9, text="(B)", space=:relative, font=:bold, fontsize=20)
text!(ax3, 0.02, 0.9, text="(C)", space=:relative, font=:bold, fontsize=20)
text!(ax4, 0.02, 0.9, text="(D)", space=:relative, font=:bold, fontsize=20)

colsize!(fig.layout, 1, Relative(0.2))
colsize!(fig.layout, 3, Relative(0.2))
colsize!(fig.layout, 2, Relative(0.28))
colsize!(fig.layout, 4, Relative(0.32))
colgap!(fig.layout, -10)

# save("2LS.pdf", fig)

fig


## double well
function double_well_pot(x)
    CM_TO_HARTREE = 4.556335252912e-6
    mass = 1836.0 # a.u.
    omega_b = 500.0  * CM_TO_HARTREE# cm-1
    v0 = 1500.0 * CM_TO_HARTREE # cm-1
    return -0.5 * mass * omega_b^2 * x^2 + mass^2 * omega_b^4/16.0 / v0 * x^4
end

temperatures = range(50.0, 400.0, step=50.0)

colors = cgrad([:blue, :red], categorical=true, length(temperatures))

fig = Figure(size=(900, 350), linewidth=2)
ax1 = Axis(fig[1,1], xlabel=L"x\, (\mathrm{a.u.})",
    ylabel=L"V(x)\,(\mathrm{a.u.})",
    ylabelpadding = -10)

testX = range(-2.75, 2.75, length=1001)
lines!(ax1, testX, double_well_pot.(testX), color="black")
text!(ax1, 0.5, 0.8, space=:relative,
    text = L"-\frac{1}{2}m\omega^2x^2 + \frac{m^2\omega^4}{16V_0}x^4",
    align=(:center, :center),
    fontsize=20,)

ax2 = Axis(fig[1,2],
    xlabel=L"t\,(\mathrm{a.u.})",
    ylabel=L"C_{xx}(t)")

for (idx, val) in enumerate(temperatures)
    f1 = readdlm("../data_dir/corr_doublewell_$(round(Int64, val))_data.txt")
    lines!(ax2, f1[:,1], f1[:,2], color=(colors[idx], 0.5), label=L"\textrm{T}=%$(round(Int,val))\, \textrm{K}")
    lines!(ax2, f1[:,1], f1[:,4], color=colors[idx], linestyle=:dash)
end
axislegend(ax2, nbanks=2, orientation=:horizontal,
    labelsize=16, framevisible=true, backgroundcolor=("white", 0.6),
    framewidth=0.5, framecolor=:gray)

colsize!(fig.layout, 1, Relative(0.35))

# save("doubleWell.pdf", fig)

fig

### spin-boson model
f1 = readdlm(joinpath(base_dir, "spin_boson_quapi", "heom_spin_boson_model2.txt"))
f2 = readdlm(joinpath(base_dir, "spin_boson_quapi", "check_spin_boson_model2.txt"))
# f2 = readdlm("/Users/kritanjanpolley/Documents/GitHub/Topological-Quantum-Computing-with-Anyons/data_dir/check_spin_boson_model.txt")

my_color = :firebrick1
f3 = readdlm(joinpath(base_dir, "spectra_heom_spin_boson_model2.txt"))
f4 = readdlm(joinpath(base_dir, "spectra_spin_boson_model2.txt"))

fig = Figure(size=(700, 350))

ax21 = Axis(fig[1,1],
    xlabel=L"\omega_c t",
    ylabel=L"\langle \sigma_z \rangle",
    xlabelsize=20, ylabelsize=20,
    xticklabelsize=20, yticklabelsize=20
)
lines!(ax21, f1[:,1], f1[:,2] .- f1[:,3], label="HEOM", color=(my_color, 0.3), linewidth=3)
lines!(ax21, f2[:,1], f2[:,3], linestyle=(:dash, :dense),
    label="Chain-Bath (Braid)", color=my_color)
xlims!(ax21, 0.0, 8.0)
axislegend(ax21, labelsize=18, patchsize = (40, 20))

fft_data = hcat(f4[:,1], f4[:,4], -f4[:,5], f3[:,2], f3[:,3])
freq, iw_ex, iw_app = get_fft(fft_data)

ax22 = Axis(fig[1,2],
    xlabelsize=20, ylabelsize=20,
    xticklabelsize=20, yticklabelsize=20,
    ylabel=L"I(\omega)",
    xlabel=L"\hbar \omega_c",
    yticklabelsvisible=false,
    xticks = LinearTicks(5)
)
lines!(ax22, freq, iw_ex, color=(my_color, 0.3), linewidth=2)
band!(ax22, freq, zeros(length(freq)), iw_ex, color = (my_color, 0.3))
lines!(ax22, freq, iw_app, color=my_color, linestyle=(:dash, :dense))
xlims!(ax22, -12.0, 12.0)
ylims!(ax22, 0.0004, nothing)

text!(ax22, (0.05, 0.7),
    text = L"$\epsilon=\omega_c, \,J=2\omega_c$\n$\lambda=\omega_c,\,\hbar\beta\omega_c=0.5$",
    space = :relative,
    fontsize = 20)

rowsize!(fig.layout, 1, Auto(0.7))

# save("spinBosonBraid.pdf", fig)

fig

#### H + H2 -> H2 + H
f1 = readdlm(joinpath(pwd(),"collinear_rate_temperature2.txt"), comments=true)
f2 = readdlm(joinpath(pwd(),"exact_rates_from_paper1.txt"), comments=true)

fig = Figure()

ax1 = Axis(fig[1, 1],
    xlabel = L"1000/T",
    ylabel = L"k(T)\,(\mathrm{cm}\,\mathrm{molecule}^{-1} \mathrm{s}^{-1})",
    yscale = log10,
    xticks = [1, 2, 3, 4, 5, 6],
    yticks = LogTicks(-3:2:5),
)

scatterlines!(ax1, 1000 ./ f1[:, 1], f1[:, 6],
    color = :black,
    marker = :circle,
    markersize = 8,
    linewidth = 2,
    label = "Computed Rates"
)

scatter!(ax1, 1000 ./ f2[:, 1], f2[:, 2],  color = :tomato,
    marker = :star6,  markersize = 15,
    label = "Published Data"
)

T_ticks = [100, 125, 150, 200, 300, 500, 1250]

ax2 = Axis(fig[1, 1],
    xaxisposition = :top,
    xlabel = L"T (\mathrm{K})",
    xticks = (1000 ./ T_ticks, string.(T_ticks)),
    yticklabelsvisible = false,
    yticksvisible = false,
    ygridvisible = false,
)

axislegend(ax1)
linkxaxes!(ax1, ax2)

# save("collinear_rate_h_h2.pdf", fig)

fig
