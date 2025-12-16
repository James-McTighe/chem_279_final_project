#pragma once
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include "basis.hpp"
#include "gaussian.hpp"
#include "basis_functions/atom.hpp"

// Structure to hold the SCF results
struct SCFState
{
    arma::mat P_alpha, P_beta;     
    arma::mat F_alpha, F_beta;     
    arma::mat C_alpha, C_beta;     
    arma::vec eps_alpha, eps_beta; 
};

// ADDED 'inline' to all function definitions below

inline ContractedGaussian create_s_contraction(Atom a)
{
    switch ( a.z_num )
    {
    case 1: return Hs(a.pos);
    case 6: return Cs(a.pos);
    case 7: return Ns(a.pos);
    case 8: return Os(a.pos);
    case 9: return Fs(a.pos);
    default:
        throw std::invalid_argument("create_s_contraction: unsupported atomic number");
    }
}

inline double boys_function(Gaussian A, Gaussian Ap, Gaussian B, Gaussian Bp)
{
    const double R = arma::norm(A.center - B.center);
    const double sigmaA = 1 / (A.alpha + Ap.alpha);
    const double sigmaB = 1 / (B.alpha + Bp.alpha);
    const double Ua = std::pow(M_PI / (A.alpha + Ap.alpha), 1.5);
    const double Ub = std::pow(M_PI / (B.alpha + Bp.alpha), 1.5);
    const double Vsquare = 1 / (sigmaA + sigmaB);
    const double T = Vsquare * R * R;

    if ( T < 1e-10 )
    {
        return Ua * Ub * std::sqrt(2 * Vsquare) * std::sqrt(2 / M_PI);
    } else
    {
        return Ua * Ub * std::sqrt(1 / (R * R)) * std::erf(std::sqrt(T));
    }
}

inline double gamma_AB(Atom A, Atom B)
{
    double gamma = 0.0;

    double dsk, dskp, dsl, dslp;

    // create new contracted Gaussians here because they can't be conveniently
    // accessed from the basis set vector
    auto A_STO3G = create_s_contraction(A);
    auto B_STO3G = create_s_contraction(B);

    for ( int k = 0; k < 3; k++ )
        for ( int kp = 0; kp < 3; kp++ )
            for ( int l = 0; l < 3; l++ )
                for ( int lp = 0; lp < 3; lp++ )
                {
                    double dsk = A_STO3G.contraction[k] * A_STO3G.norms[k];
                    double dskp = A_STO3G.contraction[kp] * A_STO3G.norms[kp];
                    double dsl = B_STO3G.contraction[l] * B_STO3G.norms[l];
                    double dslp = B_STO3G.contraction[lp] * B_STO3G.norms[lp];

                    double base = boys_function(A_STO3G.prim[k], A_STO3G.prim[kp],
                                               B_STO3G.prim[l], B_STO3G.prim[lp]);
                    gamma += dsk * dskp * dsl * dslp * base;
                }
    return gamma;
}

inline arma::mat gamma_matrix(const std::vector<Atom> & atoms)
{
    const int N = atoms.size();
    arma::mat gamma(N, N, arma::fill::zeros);
    for ( int a = 0; a < N; ++a )
    {
        for ( int b = 0; b < N; ++b )
        {
            gamma(a, b) = gamma_AB(atoms[a], atoms[b]);
        }
    }
    return gamma;
}

inline arma::mat fock_matrix(const std::vector<ContractedGaussian> & basis_set,
                             const std::vector<Atom> & atoms,
                             const arma::mat density_total,
                             const arma::mat density_spin)
{
    int N = basis_set.size();
    arma::mat fock(N, N, arma::fill::zeros);
    arma::vec atomic_density(atoms.size(), arma::fill::zeros);
    
    for ( int mu = 0; mu < N; ++mu )
    {
        atomic_density[basis_set[mu].atom_index] += density_total(mu, mu);
    }

    for ( int i = 0; i < N; ++i )
    {
        for ( int j = 0; j < N; ++j )
        {
            const auto& u = basis_set[i];
            const auto& v = basis_set[j];
            Atom A = atoms[u.atom_index];
            Atom B = atoms[v.atom_index];

            if ( i == j )
            {
                double paa = atomic_density[u.atom_index]; 
                double puu = density_spin(i, i); 
                double diag = -u.ion_term;
                diag += ((paa - A.z_star) - (puu - 0.5)) * gamma_AB(A, A);

                for ( int C = 0; C < (int)atoms.size(); ++C )
                {
                    if ( C == u.atom_index ) continue;
                    diag += (atomic_density(C) - atoms[C].z_star) * gamma_AB(A, atoms[C]);
                }
                fock(i, i) = diag;
            } else
            {
                fock(i, j) = 0.5 * (u.beta + v.beta) * contracted_overlap(u, v) -
                             density_spin(i, j) * gamma_AB(A, B);
            }
        }
    }
    return fock;
}



// helper function to generate new P matricies at each iteration of the SCF
// solver.
inline arma::mat build_spin_density(arma::mat mo_coefficients, int number_electrons)
{
    arma::mat density(arma::size(mo_coefficients), arma::fill::zeros);
    int N = density.n_cols;
    for ( int u = 0; u < N; ++u )
        for ( int v = 0; v < N; ++v )
            for ( int i = 0; i < number_electrons; ++i )
                density(u, v) += mo_coefficients(u, i) * mo_coefficients(v, i);
    return density;
}

inline SCFState solve_SCF_UHF(const std::vector<ContractedGaussian> & basis_set,
                               const std::vector<Atom> & atoms, 
                               int n_alpha, 
                               int n_beta,
                               bool logging = false,         // Added = false
                               double tol = 1e-6,           // Added = 1e-6
                               const SCFState* guess = nullptr)
{
    const int N = basis_set.size();
    SCFState current_state;

    if (guess != nullptr) {
        current_state.P_alpha = guess->P_alpha;
        current_state.P_beta = guess->P_beta;
    } else {
        current_state.P_alpha.zeros(N, N);
        current_state.P_beta.zeros(N, N);
    }

    int it = 0;
    while ( it < 500 ) 
    {
        arma::mat Ptot = current_state.P_alpha + current_state.P_beta;
        current_state.F_alpha = fock_matrix(basis_set, atoms, Ptot, current_state.P_alpha);
        current_state.F_beta = fock_matrix(basis_set, atoms, Ptot, current_state.P_beta);

        arma::eig_sym(current_state.eps_alpha, current_state.C_alpha, current_state.F_alpha);
        arma::eig_sym(current_state.eps_beta, current_state.C_beta, current_state.F_beta);

        arma::mat P_a_new = build_spin_density(current_state.C_alpha, n_alpha);
        arma::mat P_b_new = build_spin_density(current_state.C_beta, n_beta);

        double delta = std::max(arma::abs(P_a_new - current_state.P_alpha).max(),
                                arma::abs(P_b_new - current_state.P_beta).max());

        current_state.P_alpha = P_a_new;
        current_state.P_beta = P_b_new;

        if ( delta <= tol ) break;
        it++;
    }
    return current_state;
}

inline double calc_electronic_energy(arma::mat h_core, arma::mat fock_mat, arma::mat spin_density)
{
    int N = spin_density.n_cols;
    double energy = 0.0;
    for ( int u = 0; u < N; ++u )
        for ( int v = 0; v < N; ++v )
        {
            energy += spin_density(u, v) * (h_core(u, v) + fock_mat(u, v));
        }

    energy *= 0.5;

    return energy;
}
