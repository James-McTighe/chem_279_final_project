#pragma once
// standard lib imports
#include <armadillo>
#include <vector>

// header imports
#include "atom.hpp"

arma::mat hessian_matrix(arma::mat gradient, std::vector<Atom> atoms) {
    // should return second derivate of energies
    // output should also be mass corrected
    int N = atoms.size();
    arma::mat G(N, N, arma::fill::zeros);
    

    arma::mat matrix(3 * N, 3 * N);

}

arma::mat mass_dependent_coordinates(std::vector<Atom> atoms, double step_size);

arma::vec vibrational_frequencies(arma::mat mass_dep_coors, arma::mat hessian) {
    // after solving eigen values, we want to solve for frequencies and return those
}