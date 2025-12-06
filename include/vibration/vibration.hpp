#pragma once
// standard lib imports
#include <armadillo>
#include <vector>

// header imports
#include "atom.hpp"
#include "fock.hpp"
#include "gradient.hpp"
#include "hamiltonian.hpp"

// iteratively updates all atom positions and computes the second direvative for
// each using central difference
arma::mat double_central_derivative_approx(std::vector<Atom> atoms, int n_alpha,
                                           int n_beta,
                                           double step_size = 1.0e-3)
{
    size_t N = atoms.size();
    arma::mat second_dir(3 * atoms.size(), 3 * atoms.size(), arma::fill::zeros);

    for ( size_t i = 0; i < N; ++i )
    {
        for ( int j = 0; j < 3; ++j )
        {
            // initial
            std::vector<Atom> initial = atoms;
            std::vector<ContractedGaussian> initial_basis =
                make_sto3g_basis_from_xyz(initial);

            arma::mat initial_ham = core_hamiltonian(initial_basis, initial);
            SCFState initial_state =
                solve_SCF_UHF(initial_basis, initial, n_alpha, n_beta);
            double initial_alpha = calc_electronic_energy(
                initial_ham, initial_state.F_alpha, initial_state.P_alpha);
            double initial_beta = calc_electronic_energy(
                initial_ham, initial_state.F_beta, initial_state.P_beta);
            double initial_nuclear = nuclear_repulsion(initial);

            double initial_energy =
                initial_alpha + initial_beta + initial_nuclear;

            // forward
            std::vector<Atom> forward = atoms;
            forward[i].pos[j] += step_size;
            std::vector<ContractedGaussian> forward_basis =
                make_sto3g_basis_from_xyz(forward);

            arma::mat forward_ham = core_hamiltonian(forward_basis, forward);
            SCFState forward_state =
                solve_SCF_UHF(forward_basis, forward, n_alpha, n_beta);
            double forward_alpha = calc_electronic_energy(
                forward_ham, forward_state.F_alpha, forward_state.P_alpha);
            double forward_beta = calc_electronic_energy(
                forward_ham, forward_state.F_beta, forward_state.P_beta);
            double forward_nuclear = nuclear_repulsion(forward);

            double forward_energy =
                forward_alpha + forward_beta + forward_nuclear;

            // backwards
            std::vector<Atom> backward = atoms;
            backward[i].pos[j] -= step_size;
            std::vector<ContractedGaussian> backward_basis =
                make_sto3g_basis_from_xyz(backward);

            arma::mat backward_ham = core_hamiltonian(backward_basis, backward);
            SCFState backward_state =
                solve_SCF_UHF(backward_basis, backward, n_alpha, n_beta);
            double backward_alpha = calc_electronic_energy(
                backward_ham, backward_state.F_alpha, backward_state.P_alpha);
            double backward_beta = calc_electronic_energy(
                backward_ham, backward_state.F_beta, backward_state.P_beta);
            double backward_nuclear = nuclear_repulsion(backward);

            double backward_energy =
                backward_alpha + backward_beta + backward_nuclear;

            // Update second derivative
            double second_dir_element =
                (forward_energy - 2 * initial_energy + backward_energy) /
                (step_size * step_size);
        }
    }

    return second_dir;
}

arma::mat hessian_matrix(std::vector<Atom> atoms, int n_alpha, int n_beta)
{
    // should return second derivate of energies
    // output should also be mass corrected
    int N = atoms.size();

    // Calculate and set mass matrix
    arma::mat G(N, N, arma::fill::zeros);
    auto mass_element = [atoms](Atom A, Atom B) {
        return 1 / std::sqrt(A.mass * B.mass);
    };

    for ( int i = 0; i < N; ++i )
        for ( int j = 0; j < N; ++j )
        {
            G(i, j) = mass_element(atoms[i], atoms[j]);
        }

    arma::mat F(3 * N, 3 * N, arma::fill::zeros);

    // TODO fill F with second derivative values

    F = double_central_derivative_approx(atoms, n_alpha, n_beta);

    F %= G;

    return F;
}

arma::mat mass_dependent_coordinates(std::vector<Atom> atoms, double step_size);

arma::vec vibrational_frequencies(arma::mat mass_dep_coors, arma::mat hessian)
{
    // after solving eigen values, we want to solve for frequencies and return
    // those
}