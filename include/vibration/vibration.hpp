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
arma::mat double_central_derivative_approx(std::vector<Atom> atoms,
                                           const int & n_alpha,
                                           const int & n_beta,
                                           const double & step_size)
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

            // Update second derivative -> f(x + h) - f(x -h) / (2*h)
            arma::vec gradient_column =
                (forward_gradient - backward_gradient) / (2 * step_size);

            second_dir.col(column_idx) = gradient_column;
        }
    }

    return second_dir;
}

arma::vec mass_vector(const std::vector<Atom> & atoms)
{
    size_t N = 3 * atoms.size();
    arma::vec mass_vec(N, arma::fill::zeros);

    for ( int i = 0; i < N; ++i )
    {
        size_t idx = i / 3;
        double mass = atoms[idx].mass;
        mass_vec[i] = mass;
    }

    return mass_vec / 1822.888486;
}

arma::mat hessian_matrix(std::vector<Atom> atoms, const int & n_alpha,
                         const int & n_beta, const double & step_size = 1.0e-3)
{
    // should return second derivate of energies
    // output should also be mass corrected

    // Calculate and set mass matrix
    arma::vec mass_vec = mass_vector(atoms);
    int N = mass_vec.size();
    arma::mat G(N, N, arma::fill::zeros);


    for ( int i = 0; i < N; ++i )
        for ( int j = 0; j < N; ++j )
        {
            double mass_i = mass_vec[i];
            double mass_j = mass_vec[j];
            G(i, j) = 1 / std::sqrt(mass_i * mass_j);
        }

    arma::mat F(N, N, arma::fill::zeros);


    F = double_central_derivative_approx(atoms, n_alpha, n_beta, step_size);

    arma::mat dy = F;


    F %= G;


    return F;
}


// Solves the eigenvalue problem to find vibrational frequencies (omega^2)
arma::vec vibrational_frequencies(arma::mat mass_weighted_hessian,
                                  std::vector<Atom> atoms,
                                  const double & step_size)
{
    arma::vec eigenvalues;
    arma::mat eigenvectors; // Not returned, but computed alongside
    arma::mat M = arma::symmatu(mass_weighted_hessian);
    arma::eig_sym(eigenvalues, eigenvectors, M);


    arma::vec frequencies(eigenvalues.n_elem);
    for ( size_t i = 0; i < eigenvalues.n_elem; ++i )
    {
        if ( eigenvalues[i] > 0 )
        {
            // Real frequency
            frequencies[i] = (1.0 / (2.0 * M_PI)) * std::sqrt(eigenvalues[i]);
        } else
        {
            // Imaginary frequency (negative eigenvalue indicates saddle point)
            frequencies[i] = -(1.0 / (2.0 * M_PI)) * std::sqrt(-eigenvalues[i]);
        }
    }

    return frequencies;
}