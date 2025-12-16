#pragma once
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "basis.hpp"
#include "gaussian.hpp"

// from hw4 solution

// for use in gamma_AB my normal function for creating basis sets automatically
// creates p orbitals which are ignored for gamma to preserve rotational
// invariance
ContractedGaussian create_s_contraction(Atom a)
{
    switch ( a.z_num )
    {
    case 1:
        return Hs(a.pos);
    case 6:
        return Cs(a.pos);
    case 7:
        return Ns(a.pos);
    case 8:
        return Os(a.pos);
    case 9:
        return Fs(a.pos);
    default:
        // signal that the atom type is unsupported for s-contraction
        throw std::invalid_argument(
            "create_s_contraction: unsupported atomic number");
    }
}

double boys_function(Gaussian A, Gaussian Ap, Gaussian B, Gaussian Bp)
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

double gamma_AB(Atom A, Atom B)
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
                    dsk = A_STO3G.contraction[k] * A_STO3G.norms[k];
                    dskp = A_STO3G.contraction[kp] * A_STO3G.norms[kp];
                    dsl = B_STO3G.contraction[l] * B_STO3G.norms[l];
                    dslp = B_STO3G.contraction[lp] * B_STO3G.norms[lp];

                    double base =
                        boys_function(A_STO3G.prim[k], A_STO3G.prim[kp],
                                      B_STO3G.prim[l], B_STO3G.prim[lp]);
                    gamma += dsk * dskp * dsl * dslp * base * 27.211324570273;
                }
    return gamma;
}

// creates gamma matrix for all atoms
// this function isn't used for anything aside from output to the test file
arma::mat gamma_matrix(const std::vector<Atom> & atoms)
{
    const int N = atoms.size();

    arma::mat gamma(N, N, arma::fill::zeros);

    for ( int a = 0; a < N; ++a )
    {
        for ( int b = 0; b < N; ++b )
        {
            const double gij = gamma_AB(atoms[a], atoms[b]);
            gamma(a, b) = gamma(b, a) = gij;
        }
    }
    return gamma;
}

arma::mat fock_matrix(const std::vector<ContractedGaussian> & basis_set,
                      const std::vector<Atom> & atoms,
                      const arma::mat density_total,
                      const arma::mat density_spin)
{
    int N = basis_set.size();
    arma::mat fock(N, N, arma::fill::zeros);

    // vector containg total electron density for each atom
    arma::vec atomic_density(atoms.size(), arma::fill::zeros);
    for ( int mu = 0; mu < basis_set.size(); ++mu )
    {
        const int A = basis_set[mu].atom_index;
        atomic_density[A] += density_total(mu, mu);
    }

    for ( int i = 0; i < N; ++i )
        for ( int j = 0; j < N; ++j )
        {
            ContractedGaussian u = basis_set[i];
            ContractedGaussian v = basis_set[j];

            Atom A = atoms[u.atom_index];
            Atom B = atoms[v.atom_index];

            if ( i == j )
            {
                double paa = atomic_density[u.atom_index]; // total electron
                                                           // density on atom A

                double puu = density_spin(i, i); // spin density at index uu
                double diag = 0.0;
                diag -= u.ion_term;
                diag += ((paa - A.z_star) - (puu - 0.5)) * gamma_AB(A, A);

                // sum over all atoms except for A
                for ( int C = 0; C < atoms.size(); ++C )
                {
                    if ( C == u.atom_index )
                        continue;
                    diag += (atomic_density(C) - atoms[C].z_star) *
                            gamma_AB(A, atoms[C]);
                }
                fock(i, i) = diag;
            } else
            {
                fock(i, j) =
                    0.5 * (u.beta + v.beta) * contracted_overlap(u, v) -
                    density_spin(i, j) * gamma_AB(A, B);
            }
        }

    return fock;
}

// stores individual states of the SCF solution
// stores both the alpha and beta spins
struct SCFState
{
    arma::mat P_alpha, P_beta;     // AO densities
    arma::mat F_alpha, F_beta;     // Fock
    arma::mat C_alpha, C_beta;     // MO coefficients
    arma::vec eps_alpha, eps_beta; // solved eigenvalues
};

// helper function to generate new P matricies at each iteration of the SCF
// solver.  
arma::mat build_spin_density(arma::mat mo_coefficients, int number_electrons)
{
    arma::mat density(arma::size(mo_coefficients), arma::fill::zeros);
    int N = density.n_cols;

    // outer two loops iterate of MO indicies, innermost loop iterates over
    // electrons
    for ( int u = 0; u < N; ++u )
        for ( int v = 0; v < N; ++v )
            for ( int i = 0; i < number_electrons; ++i )
            {
                density(u, v) += mo_coefficients(u, i) * mo_coefficients(v, i);
            }
    return density;
}


// This function solves the SCF for Unrestricted Hartree Fock and returns it in
// a state object.  The initial density is set to zero and with each iteration a
// new fock matrix is build, the eigen values and coefficients solved, and
// convergence checked.  if the funciton has converged. The atom densities are
// updated and the current state is returned.
SCFState solve_SCF_UHF(const std::vector<ContractedGaussian> & basis_set,
                       const std::vector<Atom> & atoms, int n_alpha, int n_beta,
                       bool logging = false, double tol = 1e-6)
{
    const int N = basis_set.size();

    SCFState current_state;
    // initial guess
    current_state.P_alpha.zeros(N, N);
    current_state.P_beta.zeros(N, N);

    // helper lambda, since the basis set and atoms will always be the same.
    auto build_fock = [&](const arma::mat & Ptot, const arma::mat & Pspin) {
        return fock_matrix(basis_set, atoms, Ptot, Pspin);
    };


    int it = 0;

    if ( logging )
        std::cout << "Beginning SCF solver\n";

    while ( true )
    {
        if ( logging )
        {
            std::cout << "-------------\n";
            if ( it == 0 )
                std::cout << "Initial Iteration\n";
            else
                std::cout << "Iteration " << it << "\n";
        }
        arma::mat Ptot = current_state.P_alpha + current_state.P_beta;

        current_state.F_alpha = build_fock(Ptot, current_state.P_alpha);
        current_state.F_beta = build_fock(Ptot, current_state.P_beta);

        arma::eig_sym(current_state.eps_alpha, current_state.C_alpha,
                      current_state.F_alpha);
        arma::eig_sym(current_state.eps_beta, current_state.C_beta,
                      current_state.F_beta);

        arma::mat P_alpha_new =
            build_spin_density(current_state.C_alpha, n_alpha);
        arma::mat P_beta_new = build_spin_density(current_state.C_beta, n_beta);

        // check individual values changed as opposed to the entire matrix
        auto maxabs = [](const arma::mat & M) { return arma::abs(M).max(); };

        double delta = std::max(maxabs(P_alpha_new - current_state.P_alpha),
                                maxabs(P_beta_new - current_state.P_beta));

        current_state.P_alpha = P_alpha_new;
        current_state.P_beta = P_beta_new;

        if ( logging )
        {
            std::cout << "Fa\n" << current_state.F_alpha << std::endl;
            std::cout << "Fb\n" << current_state.F_beta << std::endl;
            std::cout << "Pa\n" << current_state.P_alpha << std::endl;
            std::cout << "Pb\n" << current_state.P_beta << std::endl;
            std::cout << "Ca\n" << current_state.C_alpha << std::endl;
            std::cout << "Cb\n" << current_state.C_beta << std::endl;
            std::cout << "Ea\n" << current_state.eps_alpha << std::endl;
            std::cout << "Eb\n" << current_state.eps_beta << std::endl;
        }

        // add the final density to each atom and break the loop
        if ( delta <= tol )
        {
            for ( int i = 0; i < N; ++i )
            {
                Atom target_atom = atoms[basis_set[i].atom_index];
                target_atom.P_total_alpha += current_state.P_alpha(i, i);
                target_atom.P_total_beta += current_state.P_beta(i, i);
            }
            break;
        }

        it++;
    }

    if ( logging )
        std::cout << "SCF complete, total iterations = " << it
                  << "\n*****************\n";
    return current_state;
}

// from equation 2.5 in instructions, this is used to calculate the first and
// second terms of the equation.  The energy output of this function corresponds
// to spin in one direction
double calc_electronic_energy(arma::mat h_core, arma::mat fock_matrix,
                              arma::mat spin_density)
{
    int N = spin_density.n_cols;
    double energy = 0.0;

    for ( int u = 0; u < N; ++u )
        for ( int v = 0; v < N; ++v )
        {
            energy += spin_density(u, v) * (h_core(u, v) + fock_matrix(u, v));
        }

    energy *= 0.5;

    return energy;
}
