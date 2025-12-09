#include "all_headers.hpp"

/**
 * A large portion of this main file was taken from the implementation for hw5
 * Some portions have been modified
 * Header files are read from a single master header in the "include" directory
 */


namespace fs = std::filesystem;
using json = nlohmann::json;

int main(int argc, char ** argv)
{
    // check that a config file is supplied
    if ( argc != 2 )
    {
        std::cerr << "\nUsage: " << argv[0] << " path/to/config.json\n\n";
        return EXIT_FAILURE;
    }
    // parse the config file
    fs::path config_file_path(argv[1]);
    if ( !fs::exists(config_file_path) )
    {
        std::cerr << "\nPath: " << config_file_path << " does not exist\n\n";
        return EXIT_FAILURE;
    }
    std::ifstream config_file(config_file_path);
    json config = json::parse(config_file);

    // extract the important info from the config file
    fs::path atoms_file_path = config["atoms_file_path"];
    fs::path output_file_path = config["output_file_path"];
    int num_alpha_electrons = config["num_alpha_electrons"];
    int num_beta_electrons = config["num_beta_electrons"];


    std::vector<Atom> atoms = parse_file(atoms_file_path);
    std::vector<ContractedGaussian> basis_set =
        make_sto3g_basis_from_xyz(atoms);


    int num_atoms = atoms.size();
    int num_basis_functions = basis_set.size();
    int num_3D_dims = 3;

    // Your answers go in these objects
    // Information about the convention of the requirements
    std::cout << "Order of columns for Suv_RA is as follows: (u,v)"
              << std::endl;
    for ( int u = 0; u < num_basis_functions; u++ )
    {
        for ( int v = 0; v < num_basis_functions; v++ )
        {
            std::cout << std::format("({},{}) ", u, v);
        }
    }
    std::cout << std::endl;

    std::cout << "Order of columns for gammaAB_RA is as follows: (A,B)"
              << std::endl;
    for ( int A = 0; A < num_atoms; A++ )
    {
        for ( int B = 0; B < num_atoms; B++ )
        {
            std::cout << std::format("({},{}) ", A, B);
        }
    }
    std::cout << std::endl;
    std::cout << "Order of rows is as follows" << std::endl;
    std::cout << "x" << std::endl;
    std::cout << "y" << std::endl;
    std::cout << "z" << std::endl;

    /**
     * Initialize Vectors
     * Based on input and a standard 3-dimensions
     */

    arma::mat Suv_RA(num_3D_dims, num_basis_functions * num_basis_functions);
    Suv_RA.zeros();
    arma::mat gammaAB_RA(num_3D_dims, num_atoms * num_atoms);
    arma::mat gradient_nuclear(num_3D_dims, num_atoms);
    arma::mat gradient_electronic(num_3D_dims, num_atoms);
    arma::mat gradient(num_3D_dims, num_atoms);

    // most of your code will go here

    // solve scf and get final state
    arma::mat H_core = core_hamiltonian(basis_set, atoms);
    SCFState final_SCF_state = solve_SCF_UHF(
        basis_set, atoms, num_alpha_electrons, num_beta_electrons);

    // Nuclear Gradient
    gradient_nuclear = nuclear_repulsion_gradient(atoms);

    // X Matrix
    arma::mat x = x_matrix(basis_set, final_SCF_state);

    // Suv_RA
    for ( int u = 0; u < num_basis_functions; ++u )
    {
        for ( int v = 0; v < num_basis_functions; ++v )
        {
            if ( basis_set[u].atom_index == basis_set[v].atom_index )
                continue;
            int col = u * num_basis_functions + v;
            Suv_RA.col(col) = overlap_gradient(basis_set[u], basis_set[v]);
        }
    }


    // gammaAB_RA
    arma::mat y = y_matrix(basis_set, atoms, final_SCF_state);

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


    gradient_electronic.zeros();

    // First electronic term: sum_(u!=v)
    for ( int u = 0; u < num_basis_functions; ++u )
    {
        int A_idx = basis_set[u].atom_index;

        for ( int v = 0; v < num_basis_functions; ++v )
        {
            int B_idx = basis_set[v].atom_index;
            if ( u == v )
                continue;

            int col = u * num_basis_functions + v;
            gradient_electronic.col(A_idx) += x(u, v) * Suv_RA.col(col);
        }
    }

    // Second electronic term: sum_(B!=A)
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

    gradient = gradient_electronic + gradient_nuclear;

    /**
     * Vibration Calculations
     * TODO Geometry Optimization
     * TODO Hessian Matrix
     * TODO Mass weighting and diagonalization
     */

    // Set print configs
    std::cout << std::fixed << std::setprecision(4) << std::setw(8)
              << std::right;

    Suv_RA.print("Suv_RA");
    gammaAB_RA.print("gammaAB_RA");
    gradient_nuclear.print("gradient_nuclear");
    gradient_electronic.print("gradient_electronic");
    gradient.print("gradient");

    // check that output dir exists
    if ( !fs::exists(output_file_path.parent_path()) )
    {
        fs::create_directories(output_file_path.parent_path());
    }

    // delete the file if it does exist (so that no old answers stay there by
    // accident)
    if ( fs::exists(output_file_path) )
    {
        fs::remove(output_file_path);
    }

    // write results to file
    Suv_RA.save(
        arma::hdf5_name(output_file_path, "Suv_RA",
                        arma::hdf5_opts::append + arma::hdf5_opts::trans));
    gammaAB_RA.save(
        arma::hdf5_name(output_file_path, "gammaAB_RA",
                        arma::hdf5_opts::append + arma::hdf5_opts::trans));
    gradient_nuclear.save(
        arma::hdf5_name(output_file_path, "gradient_nuclear",
                        arma::hdf5_opts::append + arma::hdf5_opts::trans));
    gradient_electronic.save(
        arma::hdf5_name(output_file_path, "gradient_electronic",
                        arma::hdf5_opts::append + arma::hdf5_opts::trans));
    gradient.save(
        arma::hdf5_name(output_file_path, "gradient",
                        arma::hdf5_opts::append + arma::hdf5_opts::trans));

    // TODO write IR solutions
}