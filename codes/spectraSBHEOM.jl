using QuantumToolbox
using HierarchicalEOM
using LinearAlgebra
using SparseArrays
using Printf

start_time = time()

data_dir = "../data_dir"
get_peak_memory_bytes() = Sys.maxrss()

@printf("Initial Peak: %.5f GB\n", get_peak_memory_bytes() / (1024^3))

model = length(ARGS) >= 1 ? strip(ARGS[1]) : "model2"
println("Computing spectra for $(model)")
epsilon = 1.0
jval = 2.0
mu12 = 1.0
mu13 = -0.2
tmax = model == "model2" ? 8.0 : 150.0
time_step = 0.05

omega_c = 1.0
lamda = model == "model2" ? 2.0 * omega_c : 0.125 * omega_c
kT = model == "model2" ? 2.0 * omega_c : 0.2 * omega_c

Hsys = Qobj(sparse([
    0.0     0.0     0.0;
    0.0     epsilon jval;
    0.0     jval    -epsilon
]))

Q = Qobj(sparse([
    0.0     0.0     0.0;
    0.0     1.0     0.0;
    0.0     0.0    -1.0
]))

N = 4
bath = Boson_DrudeLorentz_Pade(Q, lamda, omega_c, kT, N)
tier = 6
L = M_Boson(Hsys, tier, bath)

rho0 = ket2dm(basis(3, 0))
mu0 = Qobj([
    0.0   mu12  mu13;
    mu12  0.0   0.0;
    mu13  0.0   0.0
])
rho_mu0 = rho0 * mu0

n_ados = length(L.hierarchy.idx2nvec)
ados0_mu = ADOs(rho_mu0, n_ados)
time_arr = range(0.0, tmax, step=time_step)
ados_mu_evolution = HEOMsolve(L, ados0_mu, time_arr).ados

dipole_corr = zeros(ComplexF64, length(time_arr))

for (i, ados) in enumerate(ados_mu_evolution)
    dipole_corr[i] = dot(mu0.data', ados[1].data)
end

open(joinpath(data_dir, "spectra_heom_spin_boson_$(model).txt"), "w") do f
    for (i, t) in enumerate(time_arr)
        @printf(f, "%12.6f  %20.12f  %20.12f\n",
            t, real(dipole_corr[i]), imag(dipole_corr[i]))
    end
end

@printf("Memory at the end: %.5f GB\n", get_peak_memory_bytes() / (1024^3))
@printf("It took %.5f seconds\n", time() - start_time)
