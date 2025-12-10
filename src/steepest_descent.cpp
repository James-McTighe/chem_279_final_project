#include "all_headers.hpp"
#include <iomanip>


namespace fs = std::filesystem;
using json = nlohmann::json;

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
    double E_nuclear = nuclear_repulsion(atoms);
    
    return E_electronic + E_nuclear;
}

// Wrapper for calculate_gradient to match Python bindings signature
arma::mat calculate_gradient(std::vector<Atom>& atoms,
                             std::vector<ContractedGaussian>& basis_set,
                             int num_alpha_electrons,
                             int num_beta_electrons)
{
    
    arma::vec gradient_vec = calculate_gradient(atoms, num_alpha_electrons, num_beta_electrons);
    // Reshape to matrix form (3 x num_atoms)
    return arma::reshape(gradient_vec, 3, atoms.size());
}

// Line search using golden section method to find optimal step size
double line_search(std::vector<Atom>& atoms,
                  const arma::mat& gradient,
                  std::vector<ContractedGaussian>& basis_set,
                  int num_alpha_electrons,
                  int num_beta_electrons,
                  double max_step = 0.5)
{
    // Store original positions
    std::vector<arma::vec3> original_positions;
    for (const auto& atom : atoms)
        original_positions.push_back(atom.pos);
    
    // Get initial energy before taking a step
    double initial_energy = calculate_total_energy(atoms, basis_set, num_alpha_electrons, num_beta_electrons);
    
    // Go in negative gradient direction
    arma::mat search_dir = -gradient;
    
    // Scale the max_step based on gradient magnitude to prevent huge steps
    double grad_norm = arma::norm(search_dir, "fro");
    double adaptive_max_step = std::min(max_step, 0.01 / (grad_norm + 1e-10));
    
    // Create lambda function for golden section search
    auto energy_func = [&](double step) -> double {
        // Create temporary copies to avoid modifying atoms during search
        std::vector<Atom> temp_atoms = atoms;
        for (size_t i = 0; i < temp_atoms.size(); ++i)
        {
            temp_atoms[i].pos = original_positions[i] + step * search_dir.col(i);
        }
        
        // Rebuild basis set with new positions
        auto temp_basis_set = make_sto3g_basis_from_xyz(temp_atoms);
        
        // Calculate energy
        return calculate_total_energy(temp_atoms, temp_basis_set, num_alpha_electrons, num_beta_electrons);
    };
    
    // Use Golden section search from homework
    Golden golden(1e-5);
    Bracketmethod bracket;
    
    // Bracket the minimum starting from 0 to adaptive_max_step
    bracket.bracket(0.0, adaptive_max_step, energy_func);
    
    // Find the minimum
    double optimal_step = golden.minimize(energy_func);
    
    // Set atoms to optimal positions
    for (size_t i = 0; i < atoms.size(); ++i)
    {
        atoms[i].pos = original_positions[i] + optimal_step * search_dir.col(i);
    }
    basis_set = make_sto3g_basis_from_xyz(atoms);
    
    // Verify energy decreased
    double final_energy = calculate_total_energy(atoms, basis_set, num_alpha_electrons, num_beta_electrons);
    if (final_energy > initial_energy)
    {
        // Line search failed - restore original positions
        for (size_t i = 0; i < atoms.size(); ++i)
        {
            atoms[i].pos = original_positions[i];
        }
        basis_set = make_sto3g_basis_from_xyz(atoms);
        return 0.0;
    }
    
    return optimal_step;
}

// Main steepest descent optimization routine
void steepest_descent_optimization(std::vector<Atom>& atoms,
                                   std::vector<ContractedGaussian>& basis_set,
                                   int num_alpha_electrons,
                                   int num_beta_electrons,
                                   double gradient_tol = 1e-4,
                                   int max_iterations = 100,
                                   const std::string& output_path = "",
                                   bool logging = true)
{
    if (logging)
    {
        std::cout << "\n========================================\n";
        std::cout << "Starting Steepest Descent Optimization\n";
        std::cout << "========================================\n";
        std::cout << "Convergence threshold: " << gradient_tol << " (gradient norm)\n";
        std::cout << "Maximum iterations: " << max_iterations << "\n\n";
    }
    
    std::vector<double> energies;
    std::vector<double> gradient_norms;
    
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // Calculate energy
        double energy = calculate_total_energy(atoms, basis_set, 
                                               num_alpha_electrons, 
                                               num_beta_electrons);
        
        // Calculate gradient
        arma::vec gradient_vec = calculate_gradient(atoms, num_alpha_electrons, num_beta_electrons);
        arma::mat gradient = arma::reshape(gradient_vec, 3, atoms.size());
        
        double grad_norm = arma::norm(gradient, "fro");
        
        // Store trajectory
        energies.push_back(energy);
        gradient_norms.push_back(grad_norm);
        
        // Print iteration info
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Iteration " << std::setw(3) << iter 
                  << " | Energy: " << std::setw(12) << energy
                  << " | Gradient Norm: " << std::setw(12) << grad_norm;
        
        // Check convergence
        if (grad_norm < gradient_tol)
        {
            std::cout << " | CONVERGED\n";
            
            if (logging)
            {
                std::cout << "\n========================================\n";
                std::cout << "Optimization Complete!\n";
                std::cout << "Final Energy: " << energy << " eV\n";
                std::cout << "Final Gradient Norm: " << grad_norm << "\n";
                std::cout << "Total Iterations: " << iter << "\n";
                std::cout << "========================================\n\n";
            }
            
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
                // Create output directory if it doesn't exist
                fs::path output_file_path(output_path);
                fs::path output_dir = output_file_path.parent_path();
                if (!output_dir.empty() && !fs::exists(output_dir))
                {
                    fs::create_directories(output_dir);
                }
                
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
                // Append final energy as last line
                outfile << "Final Energy: " << std::setprecision(10) << energy << " eV\n";
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
    
    // Save final geometry if max iteration reached without convergence
    if (!output_path.empty())
    {
        // Create output directory if it doesn't exist
        fs::path output_file_path(output_path);
        fs::path output_dir = output_file_path.parent_path();
        if (!output_dir.empty() && !fs::exists(output_dir))
        {
            fs::create_directories(output_dir);
        }
        
        std::ofstream outfile(output_path);
        outfile << atoms.size() << "\n";
        outfile << "Steepest Descent (NOT CONVERGED) - Energy: " 
               << std::setprecision(10) << energies.back() << " eV\n";
        for (const auto& atom : atoms)
        {
            outfile << std::setw(2) << atom.z_num << " "
                   << std::setw(15) << std::setprecision(8) << atom.pos[0] << " "
                   << std::setw(15) << std::setprecision(8) << atom.pos[1] << " "
                   << std::setw(15) << std::setprecision(8) << atom.pos[2] << "\n";
        }
        // Append final energy as last line
        outfile << "Final Energy: " << std::setprecision(10) << energies.back() << " eV (NOT CONVERGED)\n";
        outfile.close();
        std::cout << "Final geometry saved to: " << output_path << "\n";
    }
}

#ifndef NO_MAIN
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
#endif
