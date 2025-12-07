#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Only include necessary headers, not all_headers.hpp to avoid multiple definitions
#include <armadillo>
#include "basis_functions/atom.hpp"
#include "basis_functions/Gaussian.hpp"
#include "basis_functions/basis.hpp"

namespace py = pybind11;

// Forward declarations from steepest_descent.cpp
double calculate_total_energy(std::vector<Atom>& atoms, 
                              std::vector<ContractedGaussian>& basis_set,
                              int num_alpha_electrons, 
                              int num_beta_electrons);

arma::mat calculate_gradient(std::vector<Atom>& atoms,
                             std::vector<ContractedGaussian>& basis_set,
                             int num_alpha_electrons,
                             int num_beta_electrons);

void steepest_descent_optimization(std::vector<Atom>& atoms,
                                   std::vector<ContractedGaussian>& basis_set,
                                   int num_alpha_electrons,
                                   int num_beta_electrons,
                                   double gradient_tol,
                                   int max_iterations,
                                   const std::string& output_path);

// Helper function to convert Armadillo matrix to NumPy array
py::array_t<double> arma_to_numpy(const arma::mat& mat) {
    py::array_t<double> result({mat.n_rows, mat.n_cols});
    auto buf = result.request();
    double *ptr = static_cast<double*>(buf.ptr);
    
    for (size_t i = 0; i < mat.n_rows; i++) {
        for (size_t j = 0; j < mat.n_cols; j++) {
            ptr[i * mat.n_cols + j] = mat(i, j);
        }
    }
    return result;
}

// Helper function to convert NumPy array to Armadillo vector
arma::vec3 numpy_to_arma_vec3(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1 || buf.shape[0] != 3) {
        throw std::runtime_error("Input must be a 1D array of size 3");
    }
    double *ptr = static_cast<double*>(buf.ptr);
    return arma::vec3({ptr[0], ptr[1], ptr[2]});
}

// Wrapper class for easier Python usage
class SteepestDescentOptimizer {
private:
    std::vector<Atom> atoms_;
    std::vector<ContractedGaussian> basis_set_;
    int num_alpha_electrons_;
    int num_beta_electrons_;
    
public:
    SteepestDescentOptimizer(const std::string& atoms_file_path,
                            int num_alpha_electrons,
                            int num_beta_electrons)
        : num_alpha_electrons_(num_alpha_electrons),
          num_beta_electrons_(num_beta_electrons)
    {
        atoms_ = parse_file(atoms_file_path);
        basis_set_ = make_sto3g_basis_from_xyz(atoms_);
    }
    
    // Set atoms from list of (atomic_number, x, y, z)
    void set_geometry(py::list atom_data) {
        atoms_.clear();
        for (auto item : atom_data) {
            auto atom_tuple = item.cast<py::tuple>();
            if (py::len(atom_tuple) != 4) {
                throw std::runtime_error("Each atom must be (z_num, x, y, z)");
            }
            int z_num = atom_tuple[0].cast<int>();
            double x = atom_tuple[1].cast<double>();
            double y = atom_tuple[2].cast<double>();
            double z = atom_tuple[3].cast<double>();
            
            Atom atom;
            atom.z_num = z_num;
            atom.pos = arma::vec3({x, y, z});
            atom.z_star = z_num; // For simplicity
            atoms_.push_back(atom);
        }
        basis_set_ = make_sto3g_basis_from_xyz(atoms_);
    }
    
    // Get current geometry as list of (z_num, x, y, z)
    py::list get_geometry() {
        py::list result;
        for (const auto& atom : atoms_) {
            result.append(py::make_tuple(atom.z_num, 
                                        atom.pos[0], 
                                        atom.pos[1], 
                                        atom.pos[2]));
        }
        return result;
    }
    
    // Calculate energy at current geometry
    double calculate_energy() {
        return calculate_total_energy(atoms_, basis_set_,
                                     num_alpha_electrons_,
                                     num_beta_electrons_);
    }
    
    // Calculate gradient at current geometry
    py::array_t<double> calculate_gradient_numpy() {
        arma::mat grad = calculate_gradient(atoms_, basis_set_,
                                           num_alpha_electrons_,
                                           num_beta_electrons_);
        return arma_to_numpy(grad);
    }
    
    // Run optimization
    void optimize(double gradient_tol = 1e-4,
                 int max_iterations = 100,
                 const std::string& output_path = "") {
        steepest_descent_optimization(atoms_, basis_set_,
                                     num_alpha_electrons_,
                                     num_beta_electrons_,
                                     gradient_tol,
                                     max_iterations,
                                     output_path);
    }
    
    // Get number of atoms
    int num_atoms() const {
        return atoms_.size();
    }
    
    // Save current geometry to XYZ file
    void save_geometry(const std::string& filename) {
        std::ofstream outfile(filename);
        outfile << atoms_.size() << "\n";
        outfile << "Geometry\n";
        for (const auto& atom : atoms_) {
            outfile << atom.z_num << " "
                   << std::setprecision(10) << atom.pos[0] << " "
                   << atom.pos[1] << " "
                   << atom.pos[2] << "\n";
        }
        outfile.close();
    }
};

PYBIND11_MODULE(steepest_descent_py, m) {
    m.doc() = "Python bindings for steepest descent geometry optimization";
    
    py::class_<SteepestDescentOptimizer>(m, "SteepestDescentOptimizer")
        .def(py::init<const std::string&, int, int>(),
             py::arg("atoms_file_path"),
             py::arg("num_alpha_electrons"),
             py::arg("num_beta_electrons"),
             "Initialize optimizer from an atoms file")
        .def("set_geometry", &SteepestDescentOptimizer::set_geometry,
             py::arg("atom_data"),
             "Set geometry from list of (z_num, x, y, z) tuples")
        .def("get_geometry", &SteepestDescentOptimizer::get_geometry,
             "Get current geometry as list of (z_num, x, y, z) tuples")
        .def("calculate_energy", &SteepestDescentOptimizer::calculate_energy,
             "Calculate total energy at current geometry")
        .def("calculate_gradient", &SteepestDescentOptimizer::calculate_gradient_numpy,
             "Calculate gradient at current geometry (returns NumPy array)")
        .def("optimize", &SteepestDescentOptimizer::optimize,
             py::arg("gradient_tol") = 1e-4,
             py::arg("max_iterations") = 100,
             py::arg("output_path") = "",
             "Run steepest descent optimization")
        .def("num_atoms", &SteepestDescentOptimizer::num_atoms,
             "Get number of atoms")
        .def("save_geometry", &SteepestDescentOptimizer::save_geometry,
             py::arg("filename"),
             "Save current geometry to XYZ file");
}
