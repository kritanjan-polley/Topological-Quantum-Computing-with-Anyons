import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyscf import gto, scf, fci, ao2mo, symm

data_dir = "data_dir"

def compute_point(args):
    iR, R, basis = args

    mol = gto.M(
        atom=f"H 0 0 {-R/2}; H 0 0 {R/2}",
        basis=basis,
        unit="Bohr",
        charge=0,
        spin=0,
        symmetry="D2h",
    )

    mf = scf.RHF(mol)
    mf.max_memory = 12_000  # MB per process
    mf.kernel()

    if not mf.converged:
        raise RuntimeError(f"RHF failed at R = {R:.8f}")

    nmo = mf.mo_coeff.shape[1]
    irrep_names = list(
        symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
    )

    def fci_rdm1(wfnsym):
        solver = fci.FCI(mf)
        solver.wfnsym = wfnsym
        solver.spin = 0
        solver.max_memory = 12_000
        solver.conv_tol = 1e-12

        energy, ci = solver.kernel()
        dm1 = solver.make_rdm1(ci, nmo, mol.nelectron)
        return 0.5 * (dm1 + dm1.T.conj())

    states = {irrep: fci_rdm1(irrep) for irrep in ("Ag", "B1u", "B3u", "B2u")}
    dm1_sa = sum(states[irrep] for irrep in states) / len(states)

    def top_natural_orbital(target_irreps):
        inds = [i for i, name in enumerate(irrep_names) if name in target_irreps]
        if not inds:
            raise RuntimeError(f"No orbitals for {target_irreps} at R={R:.8f}")

        block = dm1_sa[np.ix_(inds, inds)]
        occs, vecs = np.linalg.eigh(block)
        idx = np.argmax(occs)
        orbital = mf.mo_coeff[:, inds] @ vecs[:, idx]
        return orbital, float(np.real_if_close(occs[idx]))

    c_sg, occ_sg = top_natural_orbital(("Ag", "A1g"))
    c_su, occ_su = top_natural_orbital(("B1u", "A1u"))
    c_px, occ_px = top_natural_orbital(("B3u", "E1ux"))
    c_py, occ_py = top_natural_orbital(("B2u", "E1uy"))

    C = np.column_stack([c_sg, c_su, c_px, c_py])
    S = mf.get_ovlp()

    for i in range(4):
        norm = np.sqrt(np.real(np.vdot(C[:, i], S @ C[:, i])))
        C[:, i] /= norm

    return iR, R, C


def export_h2_spin_orbitals(
    R_min=0.5,
    R_max=20.0,
    N_R=256,
    basis="aug-cc-pv5z",
    tol=1e-12,
    max_workers=2,
):
    start_time = time.time()
    R_grid = np.linspace(R_min, R_max, N_R)
    nsp = 4

    print(f"Generating matrix elements using {basis} across {max_workers} workers")

    tasks = [(iR, R, basis) for iR, R in enumerate(R_grid, 1)]
    results_map = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_point, task) for task in tasks]
        for future in as_completed(futures):
            iR, R, C = future.result()
            results_map[iR] = (R, C)
            print(f"Completed [{len(results_map):4d}/{N_R}], time: {time.time()-start_time:.4f} seconds",
                flush=True)

    print(f"Parallel execution took {time.time() - start_time:.4f} seconds")

    files = {
        "h1": open(os.path.join(data_dir, "h2_matrix_elements.txt"), "w", buffering=1),
        "U": open(os.path.join(data_dir, "U_matrix_elements.txt"), "w", buffering=1),
        "h4": open(os.path.join(data_dir, "h4_matrix_elements.txt"), "w", buffering=1),
        "mux": open(os.path.join(data_dir, "mux_matrix_elements.txt"), "w", buffering=1),
        "muy": open(os.path.join(data_dir, "muy_matrix_elements.txt"), "w", buffering=1),
        "muz": open(os.path.join(data_dir, "muz_matrix_elements.txt"), "w", buffering=1),
        "Enuc": open(os.path.join(data_dir, "Enuc_matrix_elements.txt"), "w", buffering=1),
    }

    mol_prev = None
    C_prev = None

    try:
        for iR in range(1, N_R + 1):
            R, C = results_map[iR]

            mol = gto.M(
                atom=f"H 0 0 {-R/2}; H 0 0 {R/2}",
                basis=basis, unit="Bohr",
                charge=0, spin=0,
                symmetry="D2h",
            )
            mf = scf.RHF(mol)
            mf.kernel()

            if C_prev is not None:
                S_cross = gto.mole.intor_cross("int1e_ovlp", mol_prev, mol)
                overlaps = C_prev.T.conj() @ S_cross @ C
                for i in range(nsp):
                    z = overlaps[i, i]
                    if abs(z) > 1e-12:
                        C[:, i] *= np.conj(z / abs(z))

            C_prev = C.copy()
            mol_prev = mol

            h1 = C.T.conj() @ mf.get_hcore() @ C
            h1 = 0.5 * (h1 + h1.T.conj())
            eri = ao2mo.restore(1, ao2mo.kernel(mol, C), nsp)

            r_ao = mol.intor("int1e_r", comp=3)
            mu_xyz = []
            for xyz in range(3):
                mu_comp = -(C.T.conj() @ r_ao[xyz] @ C)
                mu_comp = 0.5 * (mu_comp + mu_comp.T.conj())
                mu_xyz.append(mu_comp)
            mu_x, mu_y, mu_z = mu_xyz

            files["Enuc"].write(f"{R:.8f}  {mol.energy_nuc():.12f}\n")

            for a in range(nsp):
                for b in range(nsp):
                    hval = float(np.real_if_close(h1[a, b]))
                    muvals = {
                        "mux": float(np.real_if_close(mu_x[a, b])),
                        "muy": float(np.real_if_close(mu_y[a, b])),
                        "muz": float(np.real_if_close(mu_z[a, b])),
                    }
                    for spin in range(2):
                        p = 2 * a + spin + 1
                        q = 2 * b + spin + 1
                        if abs(hval) > tol:
                            files["h1"].write(f"{p}  {q}  {R:.8f}  {hval:.12f}\n")
                        for name, value in muvals.items():
                            if abs(value) > tol:
                                files[name].write(f"{p}  {q}  {R:.8f}  {value:.12f}\n")

            for a in range(nsp):
                for b in range(nsp):
                    J = eri[a, a, b, b]
                    K = eri[a, b, b, a]
                    for sa in range(2):
                        for sb in range(2):
                            p = 2 * a + sa + 1
                            q = 2 * b + sb + 1
                            if p <= q:
                                continue
                            Uval = J - K if sa == sb else J
                            Uval = float(np.real_if_close(Uval))
                            if abs(Uval) > tol:
                                files["U"].write(f"{p}  {q}  {R:.8f}  {Uval:.12f}\n")

            for a in range(nsp):
                for b in range(nsp):
                    for c in range(nsp):
                        for d in range(nsp):
                            val = float(np.real_if_close(eri[a, d, b, c]))
                            if abs(val) <= tol:
                                continue
                            for spin1 in range(2):
                                for spin2 in range(2):
                                    p = 2 * a + spin1 + 1
                                    q = 2 * b + spin2 + 1
                                    r = 2 * c + spin2 + 1
                                    s = 2 * d + spin1 + 1
                                    if p == q or r == s:
                                        continue
                                    files["h4"].write(f"{p}  {q}  {r}  {s}  {R:.8f}  {val:.12f}\n")

    finally:
        for file in files.values():
            file.close()

    print(f"It took {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    export_h2_spin_orbitals(
        R_min=0.5, R_max=20.0, N_R=256, basis="aug-cc-pv5z", max_workers=2,
    )
