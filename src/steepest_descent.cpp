#include "all_headers.hpp"
#include <iomanip>

namespace fs = std::filesystem;
using json = nlohmann::json;

double calculate_total_energy(std::vector<Atom>& atoms, 
                              std::vector<ContractedGaussian>& basis_set,
                              int num_alpha_electrons, 
                              int num_beta_electrons,
                              const SCFState* guess = nullptr) 
{
   
    SCFState scf_state = solve_SCF_UHF(basis_set, atoms, num_alpha_electrons, num_beta_electrons, false, 1e-6, guess);
    
    arma::mat h_core = core_hamiltonian(basis_set, atoms);
    
    double E_electronic = 0.0;
    int N = basis_set.size();
    arma::mat P_tot = scf_state.P_alpha + scf_state.P_beta;

    for (int u = 0; u < N; ++u) {
        for (int v = 0; v < N; ++v) {
            
            E_electronic += 0.5 * (P_tot(u, v) * h_core(u, v) + 
                                   scf_state.P_alpha(u, v) * scf_state.F_alpha(u, v) + 
                                   scf_state.P_beta(u, v) * scf_state.F_beta(u, v));
        }
    }
    
    return E_electronic + nuclear_repulsion(atoms);
}

double line_search(std::vector<Atom>& atoms,
                   const arma::mat& gradient,
                   std::vector<ContractedGaussian>& basis_set,
                   int num_alpha_electrons,
                   int num_beta_electrons,
                   const SCFState& current_state)
{
    std::vector<arma::vec3> original_positions;
    for (const auto& atom : atoms) original_positions.push_back(atom.pos);
    
    arma::mat search_dir = -gradient; 
    double grad_norm = arma::norm(search_dir, "fro");

    if (grad_norm < 1e-10) return 0.0;
    search_dir /= grad_norm; 

    auto energy_func = [&](double step) -> double {
        std::vector<Atom> temp_atoms = atoms;
        for (size_t i = 0; i < temp_atoms.size(); ++i) {
            temp_atoms[i].pos = original_positions[i] + step * search_dir.col(i);
        }
        auto temp_basis_set = make_sto3g_basis_from_xyz(temp_atoms);
        return calculate_total_energy(temp_atoms, temp_basis_set, num_alpha_electrons, num_beta_electrons, &current_state);
    };

    Bracketmethod bracket;
    double ax = 0.0;
    double bx = 0.05; 
    
    try {
        bracket.bracket(ax, bx, energy_func);
        Golden golden(1e-4); 
        golden.ax = bracket.ax;
        golden.bx = bracket.bx;
        golden.cx = bracket.cx;
        
        double optimal_step = golden.minimize(energy_func);

        for (size_t i = 0; i < atoms.size(); ++i) {
            atoms[i].pos = original_positions[i] + optimal_step * search_dir.col(i);
        }
        basis_set = make_sto3g_basis_from_xyz(atoms);
        return optimal_step;
    } 
    catch (...) {
        return 0.0;
    }
}

std::string get_symbol(int z) {
    if (z == 1) return "H";
    if (z == 6) return "C";
    if (z == 7) return "N";
    if (z == 8) return "O";
    if (z == 9) return "F";
    return std::to_string(z);
}

void steepest_descent_optimization(std::vector<Atom>& atoms,
                                   std::vector<ContractedGaussian>& basis_set,
                                   int num_alpha_electrons,
                                   int num_beta_electrons,
                                   double gradient_tol = 1e-4,
                                   int max_iterations = 100,
                                   const std::string& output_path = "",
                                   bool logging = true)
{
    // Initialize energy variables at the function scope
    double initial_energy = calculate_total_energy(atoms, basis_set, num_alpha_electrons, num_beta_electrons);
    double current_energy = initial_energy;

    if (logging)
    {
        std::cout << "\n========================================\n";
        std::cout << "Starting Steepest Descent Optimization\n";
        std::cout << "========================================\n";
    }
    
    int zero_step_count = 0; 

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // Calculate gradient and capture the converged SCF state for the next step/energy calc
        auto grad_data = calculate_gradient(atoms, num_alpha_electrons, num_beta_electrons);
        arma::vec gradient_vec = grad_data.first;
        SCFState converged_state = grad_data.second; 

        arma::mat gradient = arma::reshape(gradient_vec, 3, atoms.size());
        double grad_norm = arma::norm(gradient, "fro");
        
        if (logging) {
            std::cout << std::fixed << std::setprecision(8);
            std::cout << "Iteration " << std::setw(3) << iter 
                      << " | Energy: " << std::setw(12) << current_energy
                      << " | Gradient Norm: " << std::setw(12) << grad_norm;
        }
        
        if (grad_norm < gradient_tol)
        {
            if (logging) std::cout << " | CONVERGED\n";
            break; 
        }
        
        // Perform line search to update atom positions
        double step_size = line_search(atoms, gradient, basis_set,
                                      num_alpha_electrons, num_beta_electrons, converged_state);
        
        if (logging) std::cout << " | Step: " << std::setw(10) << step_size << "\n";

        // Check for stalling
        if (step_size <= 1e-10) zero_step_count++;
        else zero_step_count = 0;

        if (zero_step_count >= 5)
        {
            if (logging) std::cout << " | Stalled: 5 consecutive zero steps. Terminating...\n";
            break;
        }

        // Update current_energy for the next iteration's log
        current_energy = calculate_total_energy(atoms, basis_set, num_alpha_electrons, num_beta_electrons, &converged_state);
    }

    // Final Summary
    double delta_E = current_energy - initial_energy;
    if (logging) {
        std::cout << "========================================\n";
        std::cout << "Optimization Complete\n";
        std::cout << "Initial Energy: " << initial_energy << "\n";
        std::cout << "Final Energy:   " << current_energy << "\n";
        std::cout << "Delta E:        " << delta_E << "\n";
        std::cout << "========================================\n";
    }

    // Output to XYZ file
    if (!output_path.empty())
    {
        fs::path output_file_path(output_path);
        fs::path output_dir = output_file_path.parent_path();
        if (!output_dir.empty() && !fs::exists(output_dir)) fs::create_directories(output_dir);
        
        std::ofstream outfile(output_path);
        if (outfile.is_open()) {
            outfile << atoms.size() << "\n";
            // Comment line containing energy info
            outfile << "Initial Energy: " << std::fixed << std::setprecision(10) << initial_energy 
                    << " | Final Energy: " << current_energy << "\n";
            
            for (const auto& atom : atoms)
            {
                // Use get_symbol(z_num) to output "C", "H", etc.
                outfile << std::left << std::setw(2) << get_symbol(atom.z_num) << " "
                        << std::right << std::fixed << std::setw(15) << std::setprecision(8) << atom.pos[0] << " "
                        << std::setw(15) << std::setprecision(8) << atom.pos[1] << " "
                        << std::setw(15) << std::setprecision(8) << atom.pos[2] << "\n";
            }
            outfile.close();
            if (logging) std::cout << "Final data saved to: " << output_path << "\n";
        }
    }
}

#ifndef NO_MAIN
int main(int argc, char** argv)
{
    if (argc != 2) return EXIT_FAILURE;
    
    fs::path config_file_path(argv[1]);
    if (!fs::exists(config_file_path)) return EXIT_FAILURE;
    
    std::ifstream config_file(config_file_path);
    json config = json::parse(config_file);
    
    std::vector<Atom> atoms = parse_file(config["atoms_file_path"]);
    std::vector<ContractedGaussian> basis_set = make_sto3g_basis_from_xyz(atoms);
    
    steepest_descent_optimization(atoms, basis_set,
                                  config["num_alpha_electrons"], config["num_beta_electrons"],
                                  config.value("gradient_tolerance", 1e-4), 
                                  config.value("max_iterations", 100),
                                  config.value("output_geometry_path", ""), true);
    
    return EXIT_SUCCESS;
}
#endif