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
    // Optional .xyz mode: ./ir_spectrum --xyz path/to/file.xyz num_alpha num_beta
    if (argc == 5 && std::string(argv[1]) == "--xyz") {
        fs::path xyz_file_path(argv[2]);
        if (!fs::exists(xyz_file_path)) {
            std::cerr << "\nPath: " << xyz_file_path << " does not exist\n\n";
            return EXIT_FAILURE;
        }
        int num_alpha_electrons = std::stoi(argv[3]);
        int num_beta_electrons = std::stoi(argv[4]);
        std::vector<Atom> atoms = parse_file(xyz_file_path.string());
        std::vector<ContractedGaussian> basis_set = make_sto3g_basis_from_xyz(atoms);
        const double step_size = 1.0e-3;
        arma::mat H = hessian_matrix(atoms, num_alpha_electrons, num_beta_electrons, step_size);
        H.print("hessian");
        arma::vec freq = vibrational_frequencies(H, atoms);
        freq.print("Frequencies");
        return EXIT_SUCCESS;
    }

    // Default: config file mode
    if (argc != 2) {
        std::cerr << "\nUsage: " << argv[0] << " path/to/config.json\n";
        std::cerr << "   or: " << argv[0] << " --xyz path/to/file.xyz num_alpha num_beta\n\n";
        return EXIT_FAILURE;
    }
    fs::path config_file_path(argv[1]);
    if (!fs::exists(config_file_path)) {
        std::cerr << "\nPath: " << config_file_path << " does not exist\n\n";
        return EXIT_FAILURE;
    }
    std::ifstream config_file(config_file_path);
    json config = json::parse(config_file);
    fs::path atoms_file_path = config["atoms_file_path"];
    fs::path output_file_path = config["output_file_path"];
    int num_alpha_electrons = config["num_alpha_electrons"];
    int num_beta_electrons = config["num_beta_electrons"];
    std::vector<Atom> atoms = parse_file(atoms_file_path);
    std::vector<ContractedGaussian> basis_set = make_sto3g_basis_from_xyz(atoms);
    int num_atoms = atoms.size();
    int num_basis_functions = basis_set.size();
    int num_3D_dims = 3;
    const double step_size = 1.0e-3;
    arma::mat H = hessian_matrix(atoms, num_alpha_electrons, num_beta_electrons, step_size);
    H.print("hessian");
    arma::vec freq = vibrational_frequencies(H, atoms);
    freq.print("Frequencies");
}
