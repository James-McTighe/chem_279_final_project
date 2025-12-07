#pragma once

#include "fock.hpp"
#include "hamiltonian.hpp"

arma::mat x_matrix(std::vector<ContractedGaussian> basis_set,
                   SCFState current_state)
{
    arma::mat total_density = current_state.P_alpha + current_state.P_beta;
    const int N = basis_set.size();
    arma::mat x(N, N);

    // loop over all AOs
    for ( int i = 0; i < N; i++ )
        for ( int j = 0; j < N; j++ )
        {
            double P = total_density(i, j);
            double beta_a = basis_set[i].beta;
            double beta_b = basis_set[j].beta;

            x(i, j) = (beta_a + beta_b) * P;
        }

    return x;
};

arma::mat build_y_matrix(std::vector<ContractedGaussian> basis_set,
                         SCFState current_state, std::vector<Atom> atoms)
{
    int N_atoms = atoms.size();
    int N_mo = basis_set.size();

    arma::mat total_density = current_state.P_alpha + current_state.P_beta;

    arma::mat y_matrix(N_atoms, N_atoms, arma::fill::zeros);

    arma::vec Ptot_AA(N_atoms, arma::fill::zeros);
    for ( int i = 0; i < N_mo; ++i )
    {
        int A = basis_set[i].atom_index;
        Ptot_AA += total_density(i, i);
    }

    for ( int A = 0; A < N_atoms; ++A )
    {
        for ( int B = 0; B < N_atoms; ++B )
        {
            double term = Ptot_AA(A) * Ptot_AA(B) -
                          atoms[B].z_star * Ptot_AA(A) -
                          atoms[A].z_star * Ptot_AA(B);
            double sumAB = 0.0;
            for ( int u = 0; u < N_mo; ++u )
                if ( basis_set[u].atom_index == A )

                    for ( int v = 0; v < N_mo; ++v )
                        if ( basis_set[v].atom_index == B )

                            sumAB += current_state.P_alpha(u, v) *
                                         current_state.P_beta(v, u) +
                                     current_state.P_beta(u, v) *
                                         current_state.P_beta(v, u);
            y_matrix(A, B) = term - sumAB;
        }
    }


    return y_matrix;
}


arma::vec3 boys_func_derivative(Gaussian A, Gaussian Ap, Gaussian B,
                                Gaussian Bp)
{
    const arma::vec3 R = A.center - B.center;
    const double R_norm = arma::norm(R);

    const double sigmaA = 1 / (A.alpha + Ap.alpha);
    const double sigmaB = 1 / (B.alpha + Bp.alpha);

    const double Ua = std::pow(M_PI / (A.alpha + Ap.alpha), 1.5);
    const double Ub = std::pow(M_PI / (B.alpha + Bp.alpha), 1.5);

    const double Vsquare = 1 / (sigmaA + sigmaB);
    const double V = std::sqrt(Vsquare);
    const double T = Vsquare * R_norm * R_norm;

    return Ua * Ub * R / (R_norm * R_norm) *
           (-std::erf(std::sqrt(T)) / R_norm +
            2 * V / std::sqrt(M_PI) * std::exp(-T));
}

arma::vec3 gamma_derivative(Atom A, Atom B)
{
    arma::vec3 gamma(arma::fill::zeros);

    double dsk, dskp, dsl, dslp;

    // I create new contracted Gaussians here because I can't conveniently
    // access them from the basis set vector
    auto A_STO3G = create_s_contraction(A);
    auto B_STO3G = create_s_contraction(B);

    for ( int k = 0; k < 3; k++ )
        for ( int kp = 0; kp < 3; kp++ )
            for ( int l = 0; l < 3; l++ )
                for ( int lp = 0; lp < 3; lp++ )
                {
                    dsk = A_STO3G.contraction[k] * A_STO3G.norms[k];
                    dskp = A_STO3G.contraction[kp] * A_STO3G.norms[kp];
                    dsl = B_STO3G.contraction[l] * B_STO3G.norms[l];
                    dslp = B_STO3G.contraction[lp] * B_STO3G.norms[lp];

                    arma::vec3 base =
                        boys_func_derivative(A_STO3G.prim[k], A_STO3G.prim[kp],
                                             B_STO3G.prim[l], B_STO3G.prim[lp]);
                    gamma += dsk * dskp * dsl * dslp * base * 27.211324570273;
                }
    return gamma;
};

double overlap_derivative_1D(Gaussian A, Gaussian B, int dim)
{
    double result;
    double lk = A.power[dim];

    Gaussian A_term1 = A;
    A_term1.power[dim]--;

    Gaussian A_term2 = A;
    A_term2.power[dim]++;

    result = -lk * analytical_S_ab_dim(A_term1, B, dim) +
             2 * A.alpha * analytical_S_ab_dim(A_term2, B, dim);


    return result;
}


arma::vec3 overlap_gradient(const ContractedGaussian A,
                            const ContractedGaussian B)
{
    arma::vec3 grad(arma::fill::zeros);

    for ( int k = 0; k < 3; ++k )
    {
        double dAk = A.contraction[k] * A.norms[k];

        for ( int l = 0; l < 3; ++l )
        {
            double dBl = B.contraction[l] * B.norms[l];

            const Gaussian & gA = A.prim[k];
            const Gaussian & gB = B.prim[l];

            double Sx = analytical_S_ab_dim(gA, gB, 0);
            double Sy = analytical_S_ab_dim(gA, gB, 1);
            double Sz = analytical_S_ab_dim(gA, gB, 2);

            double dSx = overlap_derivative_1D(gA, gB, 0);
            double dSy = overlap_derivative_1D(gA, gB, 1);
            double dSz = overlap_derivative_1D(gA, gB, 2);

            arma::vec3 dS_kl;
            dS_kl(0) = dSx * Sy * Sz; // d/dR_Ax
            dS_kl(1) = Sx * dSy * Sz; // d/dR_Ay
            dS_kl(2) = Sx * Sy * dSz; // d/dR_Az

            grad += dAk * dBl * dS_kl;
        }
    }

    return grad;
}

// implements hw5 solution and then flattens into vector
arma::vec calculate_gradient(std::vector<Atom> atoms, int n_alpha, int n_beta)
{
    int num_atoms = atoms.size();
    std::vector<ContractedGaussian> basis = make_sto3g_basis_from_xyz(atoms);
    int num_basis_functions = basis.size();


    arma::mat Suv_RA(3, num_basis_functions * num_basis_functions);
    Suv_RA.zeros();

    arma::mat gammaAB_RA(3, num_atoms * num_atoms);

    arma::mat gradient_nuclear(3, num_atoms);
    arma::mat gradient_electronic(3, num_atoms);
    arma::mat gradient(3, num_atoms);

    // solve scf and get final state
    arma::mat S = build_overlap_matrix(basis);
    arma::mat H_core = core_hamiltonian(basis, atoms);
    SCFState final_SCF_state = solve_SCF_UHF(basis, atoms, n_alpha, n_beta);

    // Nuclear Gradient
    gradient_nuclear = nuclear_repulsion_gradient(atoms);

    // X Matrix, to be used later
    arma::mat x = x_matrix(basis, final_SCF_state);

    // Suv_RA
    for ( int u = 0; u < num_basis_functions; ++u )
    {
        for ( int v = 0; v < num_basis_functions; ++v )
        {
            if ( basis[u].atom_index == basis[v].atom_index )
                continue;
            int col = u * num_basis_functions + v;
            Suv_RA.col(col) = overlap_gradient(basis[u], basis[v]);
        }
    }


    // create y matrix
    arma::mat y = build_y_matrix(basis, final_SCF_state, atoms);

    // below loop creates the gamma matrix
    for ( int A = 0; A < num_atoms; ++A )
    {
        for ( int B = 0; B < num_atoms; ++B )
        {
            int col = A * num_atoms + B;

            if ( A == B )
            {
                gammaAB_RA.col(col).zeros();
                continue;
            }

            gammaAB_RA.col(col) = gamma_derivative(atoms[A], atoms[B]);
        }
    }


    // gardient_electronic
    gradient_electronic.zeros();

    // First electronic term using x and overlap: sum_(u!=v)
    for ( int u = 0; u < num_basis_functions; ++u )
    {
        int A_idx = basis[u].atom_index;

        for ( int v = 0; v < num_basis_functions; ++v )
        {
            int B_idx = basis[v].atom_index;
            if ( u == v )
                continue;


            int col = u * num_basis_functions + v;
            gradient_electronic.col(A_idx) += x(u, v) * Suv_RA.col(col);
        }
    }

    // Second electronic term using y and gamma: sum_(B!=A)
    for ( int A = 0; A < num_atoms; ++A )
    {
        for ( int B = 0; B < num_atoms; ++B )
        {
            if ( A == B )
                continue;

            int col = A * num_atoms + B;
            gradient_electronic.col(A) += y(A, B) * gammaAB_RA.col(col);
        }
    }

    // Full gradient
    gradient = gradient_electronic + gradient_nuclear;

    arma::vec gradient_vec = gradient.as_col();

    return gradient_vec;
}