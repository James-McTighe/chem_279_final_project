#pragma once
#include "atom.hpp"
#include "gradient.hpp"
#include <armadillo>
#include <vector>

// Iteratively updates all atom positions and computes the second derivative
// using central difference
arma::mat double_central_derivative_approx(std::vector<Atom> atoms,
                                           const int & n_alpha,
                                           const int & n_beta,
                                           const double & step_size)
{
    size_t N = atoms.size();
    arma::mat second_dir(3 * N, 3 * N, arma::fill::zeros);

    auto gradient_help = [n_alpha, n_beta](std::vector<Atom> perturbed_atoms) {
        return calculate_gradient(perturbed_atoms, n_alpha, n_beta);
    };

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

            // Update second derivative -> [f(x + h) - f(x - h)] / (2*h)
            arma::vec gradient_column =
                (forward_gradient - backward_gradient) / (2 * step_size);

            second_dir.col(column_idx) = gradient_column;
        }
    }

    return second_dir;
}

// Create mass vector in atomic units
arma::vec mass_vector(const std::vector<Atom> & atoms)
{
    size_t N = 3 * atoms.size();
    arma::vec mass_vec(N, arma::fill::zeros);

    for ( size_t i = 0; i < N; ++i )
    {
        size_t idx = i / 3;
        double mass = atoms[idx].mass; // Assuming mass is in amu
        mass_vec[i] = mass;
    }

    // Convert from amu to atomic units (electron masses)
    return mass_vec * 1822.888486;
}

// Compute mass-weighted Hessian matrix (F·G)
arma::mat hessian_matrix(std::vector<Atom> atoms, const int & n_alpha,
                         const int & n_beta, const double & step_size = 1.0e-3)
{
    // Calculate mass vector
    arma::vec mass_vec = mass_vector(atoms);
    int N = mass_vec.size();

    // Create G matrix (mass-weighting matrix)
    arma::mat G = mass_vec * mass_vec.t();

    G = 1 / arma::sqrt(G);

    // Calculate force constant matrix F (Hessian)
    arma::mat F =
        double_central_derivative_approx(atoms, n_alpha, n_beta, step_size);

    // Apply mass weighting: F_mass_weighted
    F %= G;

    return F;
}


// Calculate vibrational frequencies in cm^-1
arma::vec vibrational_frequencies(arma::mat mass_weighted_hessian,
                                  std::vector<Atom> atoms)
{
    arma::vec eigenvalues;
    arma::mat eigenvectors;

    // hessian scaling applied per Shaw et al
    mass_weighted_hessian /= 1.48;

    // Symmetrize
    arma::eig_sym(eigenvalues, eigenvectors, mass_weighted_hessian);
    // Convert eigenvalues to frequencies in hartree -> cm^-1
    arma::vec frequencies(eigenvalues.n_elem);
    for ( size_t i = 0; i < eigenvalues.n_elem; ++i )
    {
        if ( eigenvalues[i] > 1e-10 )
        {
            frequencies[i] =
                (1.0 / (2.0 * M_PI)) * std::sqrt(eigenvalues[i]) * 219474.63;
        } else
        {
            frequencies[i] = 0.0;
        }
    }

    return frequencies;
}
