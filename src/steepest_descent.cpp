#include "all_headers.hpp"
#include <iomanip>

/**
 * Steepest Descent Geometry Optimization
 * The algorithm:
 * 1. Calculate energy and gradient at current geometry
 * 2. Move atoms in direction of negative gradient (downhill)
 * 3. Use line search to find optimal step size
 * 4. Repeat until gradient norm falls below convergence threshold
 */

namespace fs = std::filesystem;
using json = nlohmann::json;

// Calculate total energy for a given geometry
double calculate_total_energy(std::vector<Atom>& atoms, 
                              std::vector<ContractedGaussian>& basis_set,
                              int num_alpha_electrons, 
                              int num_beta_electrons)
{
    // Run SCF to get converged density matrices
    SCFState scf_state = solve_SCF_UHF(basis_set, atoms, 
                                       num_alpha_electrons, 
                                       num_beta_electrons, 
                                       false); // no logging
    
    // Calculate core hamiltonian
    arma::mat h_core = core_hamiltonian(basis_set, atoms);
    
    // Calculate electronic energy (sum of alpha and beta contributions)
    double E_elec_alpha = calc_electronic_energy(h_core, scf_state.F_alpha, scf_state.P_alpha);
    double E_elec_beta = calc_electronic_energy(h_core, scf_state.F_beta, scf_state.P_beta);
    double E_electronic = E_elec_alpha + E_elec_beta;
    
    // Calculate nuclear repulsion energy
    double E_nuclear = 0.0;
    int N_atoms = atoms.size();
    for (int A = 0; A < N_atoms; ++A)
    {
        for (int B = A + 1; B < N_atoms; ++B)
        {
            double R_AB = arma::norm(atoms[A].pos - atoms[B].pos);
            E_nuclear += atoms[A].z_star * atoms[B].z_star * gamma_AB(atoms[A], atoms[B]);
        }
    }
    
    return E_electronic + E_nuclear;
}

// Calculate gradient for current geometry
arma::mat calculate_gradient(std::vector<Atom>& atoms,
                             std::vector<ContractedGaussian>& basis_set,
                             int num_alpha_electrons,
                             int num_beta_electrons)
{
    int num_atoms = atoms.size();
    int num_basis_functions = basis_set.size();
    
    // Run SCF
    SCFState scf_state = solve_SCF_UHF(basis_set, atoms,
                                       num_alpha_electrons,
                                       num_beta_electrons,
                                       false);
    
    // Build gradient matrices
    arma::mat gradient_nuclear(3, num_atoms, arma::fill::zeros);
    arma::mat gradient_electronic(3, num_atoms, arma::fill::zeros);
    
    // Calculate overlap gradient matrix Suv_RA
    arma::mat Suv_RA(3, num_basis_functions * num_basis_functions, arma::fill::zeros);
    int col = 0;
    for (int u = 0; u < num_basis_functions; ++u)
    {
        for (int v = 0; v < num_basis_functions; ++v)
        {
            arma::vec3 grad = overlap_gradient(basis_set[u], basis_set[v]);
            Suv_RA.col(col) = grad;
            col++;
        }
    }
    
    // Calculate gamma derivative matrix
    arma::mat gammaAB_RA(3, num_atoms * num_atoms, arma::fill::zeros);
    col = 0;
    for (int A = 0; A < num_atoms; ++A)
    {
        for (int B = 0; B < num_atoms; ++B)
        {
            arma::vec3 grad = gamma_derivative(atoms[A], atoms[B]);
            gammaAB_RA.col(col) = grad;
            col++;
        }
    }
    
    // Calculate nuclear gradient
    for (int A = 0; A < num_atoms; ++A)
    {
        arma::vec3 grad_A(arma::fill::zeros);
        
        for (int B = 0; B < num_atoms; ++B)
        {
            if (A == B) continue;
            
            int col_AB = A * num_atoms + B;
            grad_A += atoms[A].z_star * atoms[B].z_star * gammaAB_RA.col(col_AB);
        }
        
        gradient_nuclear.col(A) = grad_A;
    }
    
    // Calculate electronic gradient
    arma::mat x = x_matrix(basis_set, scf_state);
    arma::mat y = build_y_matrix(basis_set, scf_state, atoms);
    
    // First electronic term: overlap gradient contribution
    for (int u = 0; u < num_basis_functions; ++u)
    {
        for (int v = 0; v < num_basis_functions; ++v)
        {
            int A = basis_set[u].atom_index;
            int col_uv = u * num_basis_functions + v;
            gradient_electronic.col(A) -= x(u, v) * Suv_RA.col(col_uv);
        }
    }
    
    // Second electronic term: gamma derivative contribution
    for (int A = 0; A < num_atoms; ++A)
    {
        for (int B = 0; B < num_atoms; ++B)
        {
            if (A == B) continue;
            
            int col_AB = A * num_atoms + B;
            gradient_electronic.col(A) += y(A, B) * gammaAB_RA.col(col_AB);
        }
    }
    
    return gradient_electronic + gradient_nuclear;
}

// Line search using golden section method to find optimal step size
double line_search(std::vector<Atom>& atoms,
                  const arma::mat& gradient,
                  std::vector<ContractedGaussian>& basis_set,
                  int num_alpha_electrons,
                  int num_beta_electrons,
                  double max_step = 0.1)  //
{
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0; // Golden ratio
    const double resphi = 2.0 - phi;
    
    // Store original positions
    std::vector<arma::vec3> original_positions;
    for (const auto& atom : atoms)
        original_positions.push_back(atom.pos);
    
    // Search direction is negative gradient
    arma::mat search_dir = -gradient;
    
    // Scale the max_step based on gradient magnitude to prevent huge steps
    double grad_norm = arma::norm(search_dir, "fro");
    double adaptive_max_step = std::min(max_step, 0.5 / (grad_norm + 1e-10));
    
    // Golden section search bounds
    double a = 0.0;
    double b = adaptive_max_step;
    
    // Initial interval reduction
    double tol = 1e-5;
    double c = b - (b - a) * resphi;
    double d = a + (b - a) * resphi;
    
    // Function to evaluate energy at a given step size
    auto eval_energy = [&](double step) {
        // Update positions
        for (size_t i = 0; i < atoms.size(); ++i)
        {
            atoms[i].pos = original_positions[i] + step * search_dir.col(i);
        }
        
        // Rebuild basis set with new positions
        basis_set = make_sto3g_basis_from_xyz(atoms);
        
        // Calculate energy
        return calculate_total_energy(atoms, basis_set, num_alpha_electrons, num_beta_electrons);
    };
    
    double fc = eval_energy(c);
    double fd = eval_energy(d);
    
    // Golden section iterations
    int max_line_search_iter = 50;
    int line_iter = 0;
    while (std::abs(b - a) > tol && line_iter < max_line_search_iter)
    {
        if (fc < fd)
        {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * resphi;
            fc = eval_energy(c);
        }
        else
        {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * resphi;
            fd = eval_energy(d);
        }
        line_iter++;
    }
    
    // Return optimal step size
    double optimal_step = (a + b) / 2.0;
    
    // Set atoms to optimal positions
    for (size_t i = 0; i < atoms.size(); ++i)
    {
        atoms[i].pos = original_positions[i] + optimal_step * search_dir.col(i);
    }
    basis_set = make_sto3g_basis_from_xyz(atoms);
    
    return optimal_step;
}

// Main steepest descent optimization routine
void steepest_descent_optimization(std::vector<Atom>& atoms,
                                   std::vector<ContractedGaussian>& basis_set,
                                   int num_alpha_electrons,
                                   int num_beta_electrons,
                                   double gradient_tol = 1e-4,
                                   int max_iterations = 100,
                                   const std::string& output_path = "")
{
    std::cout << "\n========================================\n";
    std::cout << "Starting Steepest Descent Optimization\n";
    std::cout << "========================================\n";
    std::cout << "Convergence threshold: " << gradient_tol << " (gradient norm)\n";
    std::cout << "Maximum iterations: " << max_iterations << "\n\n";
    
    std::vector<double> energies;
    std::vector<double> gradient_norms;
    std::vector<std::vector<arma::vec3>> geometries;
    
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // Calculate energy
        double energy = calculate_total_energy(atoms, basis_set, 
                                               num_alpha_electrons, 
                                               num_beta_electrons);
        
        // Calculate gradient
        arma::mat gradient = calculate_gradient(atoms, basis_set,
                                               num_alpha_electrons,
                                               num_beta_electrons);
        
        double grad_norm = arma::norm(gradient, "fro");
        
        // Store trajectory
        energies.push_back(energy);
        gradient_norms.push_back(grad_norm);
        std::vector<arma::vec3> current_geom;
        for (const auto& atom : atoms)
            current_geom.push_back(atom.pos);
        geometries.push_back(current_geom);
        
        // Print iteration info
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Iteration " << std::setw(3) << iter 
                  << " | Energy: " << std::setw(12) << energy
                  << " | Gradient Norm: " << std::setw(12) << grad_norm;
        
        // Check convergence
        if (grad_norm < gradient_tol)
        {
            std::cout << " | CONVERGED\n";
            std::cout << "\n========================================\n";
            std::cout << "Optimization Complete!\n";
            std::cout << "Final Energy: " << energy << " eV\n";
            std::cout << "Final Gradient Norm: " << grad_norm << "\n";
            std::cout << "Total Iterations: " << iter << "\n";
            std::cout << "========================================\n\n";
            
            // Print final geometry
            std::cout << "Optimized Geometry:\n";
            std::cout << atoms.size() << "\n";
            std::cout << "Steepest Descent Optimized Structure\n";
            for (const auto& atom : atoms)
            {
                std::cout << std::setw(2) << atom.z_num << " "
                         << std::setw(15) << std::setprecision(8) << atom.pos[0] << " "
                         << std::setw(15) << std::setprecision(8) << atom.pos[1] << " "
                         << std::setw(15) << std::setprecision(8) << atom.pos[2] << "\n";
            }
            
            // Save optimized geometry to file if output path provided
            if (!output_path.empty())
            {
                std::ofstream outfile(output_path);
                outfile << atoms.size() << "\n";
                outfile << "Steepest Descent Optimized Structure - Energy: " 
                       << std::setprecision(10) << energy << " eV\n";
                for (const auto& atom : atoms)
                {
                    outfile << std::setw(2) << atom.z_num << " "
                           << std::setw(15) << std::setprecision(8) << atom.pos[0] << " "
                           << std::setw(15) << std::setprecision(8) << atom.pos[1] << " "
                           << std::setw(15) << std::setprecision(8) << atom.pos[2] << "\n";
                }
                outfile.close();
                std::cout << "\nOptimized geometry saved to: " << output_path << "\n";
            }
            
            return;
        }
        
        // Perform line search to find optimal step size
        double step_size = line_search(atoms, gradient, basis_set,
                                      num_alpha_electrons, num_beta_electrons);
        
        std::cout << " | Step: " << std::setw(10) << step_size << "\n";
    }
    
    std::cout << "\nWARNING: Maximum iterations reached without convergence!\n";
    std::cout << "Final gradient norm: " << gradient_norms.back() << "\n";
}

int main(int argc, char** argv)
{
    // Check command line arguments
    if (argc != 2)
    {
        std::cerr << "\nUsage: " << argv[0] << " path/to/config.json\n\n";
        return EXIT_FAILURE;
    }
    
    // Parse config file
    fs::path config_file_path(argv[1]);
    if (!fs::exists(config_file_path))
    {
        std::cerr << "\nPath: " << config_file_path << " does not exist\n\n";
        return EXIT_FAILURE;
    }
    
    std::ifstream config_file(config_file_path);
    json config = json::parse(config_file);
    
    // Extract configuration
    fs::path atoms_file_path = config["atoms_file_path"];
    int num_alpha_electrons = config["num_alpha_electrons"];
    int num_beta_electrons = config["num_beta_electrons"];
    
    // Optional parameters
    double gradient_tol = config.value("gradient_tolerance", 1e-4);
    int max_iterations = config.value("max_iterations", 100);
    std::string output_geom_path = config.value("output_geometry_path", "");
    
    // Parse initial geometry
    std::vector<Atom> atoms = parse_file(atoms_file_path);
    
    std::cout << "\nInitial Geometry:\n";
    std::cout << atoms.size() << " atoms\n";
    for (const auto& atom : atoms)
    {
        std::cout << "Atom " << atom.z_num << ": ("
                 << atom.pos[0] << ", "
                 << atom.pos[1] << ", "
                 << atom.pos[2] << ")\n";
    }
    
    // Build initial basis set
    std::vector<ContractedGaussian> basis_set = make_sto3g_basis_from_xyz(atoms);
    
    // Run optimization
    steepest_descent_optimization(atoms, basis_set,
                                 num_alpha_electrons, num_beta_electrons,
                                 gradient_tol, max_iterations,
                                 output_geom_path);
    
    return EXIT_SUCCESS;
}
