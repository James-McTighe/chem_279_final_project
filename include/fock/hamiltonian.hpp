#pragma once

#include "atom.hpp"
#include "basis.hpp"
#include "fock.hpp"

// from hw4 solution

arma::mat core_hamiltonian(std::vector<ContractedGaussian> basis_set,
                           std::vector<Atom> atoms)
{
    int N = basis_set.size();
    arma::mat hamiltonian(N, N, arma::fill::zeros);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            ContractedGaussian u = basis_set[i];
            ContractedGaussian v = basis_set[j];

            Atom A = atoms[u.atom_index];
            Atom B = atoms[v.atom_index];

            if (i == j)
            {
                double diag = 0;
                diag -= u.ion_term;
                diag -= (u.z_star - 0.5) * gamma_AB(A, A);
                for (int C = 0; C < atoms.size(); ++C)
                {
                    if (C == u.atom_index)
                        continue;
                    Atom atom_C = atoms[C];
                    diag -= atom_C.z_star * gamma_AB(atom_C, A);
                }

                hamiltonian(i, j) = diag;
            }
            else
            {
                hamiltonian(i, j) =
                    0.5 * (u.beta + v.beta) * contracted_overlap(u, v);
            }
        }

    return hamiltonian;
}