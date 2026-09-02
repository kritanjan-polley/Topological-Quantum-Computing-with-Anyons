using QuantumToolbox
using HierarchicalEOM
using Printf

start_time = time()

data_dir = "data_dir"

epsilon = 1.0
J = 2.0
lamda = 1.0
Wc = 1.0
kT = Wc * 2.0

Hsys = epsilon * sigmaz() + J * sigmax()
rho0 = ket2dm(basis(2, 0));
Q = sigmaz()

P00 = ket2dm(basis(2, 0))
P11 = ket2dm(basis(2, 1))

N = 4
bath = Boson_DrudeLorentz_Pade(Q, lamda, Wc, kT, N)
tier = 10
L = M_Boson(Hsys, tier, bath)

tlist = 0:0.05:8
sol = HEOMsolve(L, rho0, tlist; e_ops=[P00, P11])
p00_e = real(sol.expect[1, :])
p11_e = real(sol.expect[2, :])


data_file = joinpath(data_dir, "heom_spin_boson_model.txt")
open(data_file, "w") do io
    for i in eachindex(tlist)
        @printf(io, "%.6e  %.10e  %.10e\n",
            tlist[i], p00_e[i], p11_e[i]
        )
    end
end

@printf("It took %.5f seconds\n", time() - start_time)
