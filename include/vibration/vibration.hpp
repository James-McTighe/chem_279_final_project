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

    // lambda function for calculating the energy at each step
    auto energy_instance =
        [n_alpha, n_beta](std::vector<Atom> atoms_instance,
                          std::vector<ContractedGaussian> basis_instance) {
            arma::mat ham_instance =
                core_hamiltonian(basis_instance, atoms_instance);
            SCFState instance_state =
                solve_SCF_UHF(basis_instance, atoms_instance, n_alpha, n_beta);
            double instance_alpha = calc_electronic_energy(
                ham_instance, instance_state.F_alpha, instance_state.P_alpha);
            double instance_beta = calc_electronic_energy(
                ham_instance, instance_state.F_beta, instance_state.P_beta);
            double instance_nuclear = nuclear_repulsion(atoms_instance);

            return instance_alpha + instance_beta + instance_nuclear;
        };

    for ( size_t i = 0; i < N; ++i )
    {
        for ( int j = 0; j < 3; ++j )
        {
            // initial -> f(x)
            std::vector<Atom> initial = atoms;
            std::vector<ContractedGaussian> initial_basis =
                make_sto3g_basis_from_xyz(initial);

            double initial_energy = energy_instance(initial, initial_basis);

            // forward -> f(x + h)
            std::vector<Atom> forward = atoms;
            forward[i].pos[j] += step_size;
            std::vector<ContractedGaussian> forward_basis =
                make_sto3g_basis_from_xyz(forward);

            double forward_energy = energy_instance(forward, forward_basis);

            // backwards -> f(x - h)
            std::vector<Atom> backward = atoms;
            backward[i].pos[j] -= step_size;
            std::vector<ContractedGaussian> backward_basis =
                make_sto3g_basis_from_xyz(backward);

            double backward_energy = energy_instance(backward, backward_basis);

            // Update second derivative -> f(x + h) - 2 * f(x) + f(x - h) / h^2
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