#pragma once

#include "fock.hpp"
#include "hamiltonian.hpp"
#include <utility> 

/**
 * Calculates the X matrix (electronic coupling) for the gradient.
 * X_uv = (beta_A + beta_B) * P_uv
 */
arma::mat x_matrix(std::vector<ContractedGaussian> basis_set,
                   SCFState current_state)
{
    arma::mat total_density = current_state.P_alpha + current_state.P_beta;
    const int N = basis_set.size();
    arma::mat x(N, N);

    for ( int i = 0; i < N; i++ )
        for ( int j = 0; j < N; j++ )
        {
            double P = total_density(i, j);
            double beta_a = basis_set[i].beta;
            double beta_b = basis_set[j].beta;
            
            x(i, j) = ((beta_a + beta_b)) * P;
        }

    return x;
}

/**
 * Evaluates y_ab for two atoms A and B based on atomic populations.
 */
double y_ab(std::vector<ContractedGaussian> basis_set, SCFState current_state,
            Atom A, Atom B)
{
    double total_density_atom_A = 0.0;
    double total_density_atom_B = 0.0;

    for ( int mu = 0; mu < basis_set.size(); ++mu )
    {
        int atom_index = basis_set[mu].atom_index;
        if ( atom_index == A.vector_idx )
        {
            total_density_atom_A += current_state.P_alpha(mu, mu) + current_state.P_beta(mu, mu);
        }
        if ( atom_index == B.vector_idx )
        {
            total_density_atom_B += current_state.P_alpha(mu, mu) + current_state.P_beta(mu, mu);
        }
    }

    double result = total_density_atom_A * total_density_atom_B;
    result -= B.z_star * total_density_atom_A;
    result -= A.z_star * total_density_atom_B;

    for ( int u = 0; u < basis_set.size(); ++u )
    {
        if ( basis_set[u].atom_index != A.vector_idx ) continue;
        for ( int v = 0; v < basis_set.size(); ++v )
        {
            if ( basis_set[v].atom_index != B.vector_idx ) continue;
            result -= current_state.P_alpha(u, v) * current_state.P_alpha(v, u) +
                      current_state.P_beta(u, v) * current_state.P_beta(v, u);
        }
    }
    return result;
}

arma::mat y_matrix(std::vector<ContractedGaussian> basis_set,
                   std::vector<Atom> atoms, SCFState current_state)
{
    int N = atoms.size();
    arma::mat y(N, N, arma::fill::zeros);
    for ( int A = 0; A < N; ++A )
        for ( int B = 0; B < N; ++B )
            y(A, B) = y_ab(basis_set, current_state, atoms[A], atoms[B]);
    return y;
}

arma::vec3 boys_func_derivative(Gaussian A, Gaussian Ap, Gaussian B, Gaussian Bp)
{
    const arma::vec3 R_vec = A.center - B.center;
    const double R_norm = arma::norm(R_vec);
    if (R_norm < 1e-10) return arma::vec3(arma::fill::zeros);

    const double sigmaA = 1 / (A.alpha + Ap.alpha);
    const double sigmaB = 1 / (B.alpha + Bp.alpha);
    const double Ua = std::pow(M_PI / (A.alpha + Ap.alpha), 1.5);
    const double Ub = std::pow(M_PI / (B.alpha + Bp.alpha), 1.5);
    const double Vsquare = 1 / (sigmaA + sigmaB);
    const double V = std::sqrt(Vsquare);
    const double T = Vsquare * R_norm * R_norm;

    return Ua * Ub * R_vec / (R_norm * R_norm) *
           (-std::erf(std::sqrt(T)) / R_norm + 2 * V / std::sqrt(M_PI) * std::exp(-T));
}

arma::vec3 gamma_derivative(Atom A, Atom B)
{
    arma::vec3 grad(arma::fill::zeros);
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

                    grad += dsk * dskp * dsl * dslp * boys_func_derivative(A_STO3G.prim[k], A_STO3G.prim[kp], B_STO3G.prim[l], B_STO3G.prim[lp]);
                }
    return grad;
}

double overlap_derivative_1D(Gaussian A, Gaussian B, int dim)
{
    double lk = A.power[dim];
    Gaussian A_minus = A; A_minus.power[dim]--;
    Gaussian A_plus = A;  A_plus.power[dim]++;

    return -lk * analytical_S_ab_dim(A_minus, B, dim) + 2 * A.alpha * analytical_S_ab_dim(A_plus, B, dim);
}

arma::vec3 overlap_gradient(const ContractedGaussian A, const ContractedGaussian B)
{
    arma::vec3 grad(arma::fill::zeros);
    for ( int k = 0; k < 3; ++k ) {
        for ( int l = 0; l < 3; ++l ) {
            const Gaussian & gA = A.prim[k];
            const Gaussian & gB = B.prim[l];
            double Sx = analytical_S_ab_dim(gA, gB, 0);
            double Sy = analytical_S_ab_dim(gA, gB, 1);
            double Sz = analytical_S_ab_dim(gA, gB, 2);

            arma::vec3 dS_kl;
            dS_kl(0) = overlap_derivative_1D(gA, gB, 0) * Sy * Sz;
            dS_kl(1) = Sx * overlap_derivative_1D(gA, gB, 1) * Sz;
            dS_kl(2) = Sx * Sy * overlap_derivative_1D(gA, gB, 2);

            grad += (A.contraction[k] * A.norms[k]) * (B.contraction[l] * B.norms[l]) * dS_kl;
        }
    }
    return grad;
}

/**
 * Main Gradient Calculation
 * Returns {Gradient Vector, Converged SCF State}
 */
std::pair<arma::vec, SCFState> calculate_gradient(std::vector<Atom> atoms, int n_alpha, int n_beta)
{
    int num_atoms = atoms.size();
    std::vector<ContractedGaussian> basis = make_sto3g_basis_from_xyz(atoms);
    int num_basis = basis.size();

    arma::mat gradient_nuclear = nuclear_repulsion_gradient(atoms);
    arma::mat gradient_electronic(3, num_atoms, arma::fill::zeros);

    // Solve SCF to get the density matrices (P) needed for the gradient
    SCFState final_state = solve_SCF_UHF(basis, atoms, n_alpha, n_beta, false);

    arma::mat x = x_matrix(basis, final_state);
    arma::mat y = y_matrix(basis, atoms, final_state);

    // Electronic Gradient: Overlap Term
    for ( int u = 0; u < num_basis; ++u ) {
        int A_idx = basis[u].atom_index;
        for ( int v = 0; v < num_basis; ++v ) {
            if ( basis[u].atom_index == basis[v].atom_index ) continue;
            gradient_electronic.col(A_idx) += x(u, v) * overlap_gradient(basis[u], basis[v]);
        }
    }

    // Electronic Gradient: Gamma Term
    for ( int A = 0; A < num_atoms; ++A ) {
        for ( int B = 0; B < num_atoms; ++B ) {
            if ( A == B ) continue;
            gradient_electronic.col(A) += y(A, B) * gamma_derivative(atoms[A], atoms[B]);
        }
    }

    arma::mat total_gradient = gradient_electronic + gradient_nuclear;
    
    return {total_gradient.as_col(), final_state};
}