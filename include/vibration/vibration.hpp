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
    size_t total_coors = 3 * N;
    arma::mat second_dir(3 * atoms.size(), 3 * atoms.size(), arma::fill::zeros);

    auto gradient_help = [n_alpha, n_beta](std::vector<Atom> perturbed_atoms) {
        return calculate_gradient(perturbed_atoms, n_alpha, n_beta);
    };

    arma::vec gradient_base = gradient_help(atoms);

    for ( size_t i = 0; i < N; ++i )
    {
        for ( int j = 0; j < 3; ++j )
        {
            size_t column_idx = i * 3 + j;
            // forward -> f(x + h)
            std::vector<Atom> forward = atoms;
            forward[i].pos[j] += step_size;

            arma::vec forward_gradient = gradient_help(forward);

            // backwards -> f(x - h)
            std::vector<Atom> backward = atoms;
            backward[i].pos[j] -= step_size;

            arma::vec backward_gradient = gradient_help(backward);

            // Update second derivative -> f(x + h) - 2 * f(x) + f(x - h) / h^2
            arma::vec gradient_column =
                (forward_gradient - backward_gradient) / (2 * step_size);
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

arma::mat mass_dependent_coordinates(std::vector<Atom> atoms, double step_size)
{
    size_t N = atoms.size();

    arma::mat mass_weighted_coords(3 * N, 3 * N, arma::fill::zeros);


    return mass_weighted_coords;
};

arma::vec vibrational_frequencies(arma::mat mass_dep_coors, arma::mat hessian)
{
    // after solving eigen values, we want to solve for frequencies and return
    // those
}