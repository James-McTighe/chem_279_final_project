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
                                   const std::string& output_path,
                                   bool logging);

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

// Wrapper class
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
                 const std::string& output_path = "",
                 bool logging = false) {
        steepest_descent_optimization(atoms_, basis_set_,
                                     num_alpha_electrons_,
                                     num_beta_electrons_,
                                     gradient_tol,
                                     max_iterations,
                                     output_path,
                                     logging);
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
    m.doc() = R"pbdoc(
        Steepest Descent Geometry Optimization
        =======================================
        This module provides tools for optimizing molecular geometries by 
        minimizing the total energy using gradient-based steepest descent with 
        line search.
        
        Class:
            SteepestDescentOptimizer: Main optimizer class for molecular geometry optimization
    )pbdoc";
    
    py::class_<SteepestDescentOptimizer>(m, "SteepestDescentOptimizer", R"pbdoc(
        Steepest Descent Geometry Optimizer
        
        A class for optimizing molecular geometries using the steepest descent 
        method.

        Attributes:
            All attributes are internal and managed by the class.
        
        Units:
            - Coordinates: Bohr
            - Energies: eV
            - Gradients: eV/Bohr
    )pbdoc")
        .def(py::init<const std::string&, int, int>(),
             py::arg("atoms_file_path"),
             py::arg("num_alpha_electrons"),
             py::arg("num_beta_electrons"),
             R"pbdoc(
                Initialize optimizer from an atoms file.
                
                Args:
                    atoms_file_path (str): Path to XYZ file containing atomic positions.
                                          File should have standard XYZ format:
                                          line 1: number of atoms
                                          line 2: comment line
                                          remaining: atomic_number x y z (in Bohr)
                    num_alpha_electrons (int): Number of alpha spin electrons
                    num_beta_electrons (int): Number of beta spin electrons
                
                Returns:
                    SteepestDescentOptimizer: Initialized optimizer object
                
                Example:
                    >>> optimizer = SteepestDescentOptimizer("water.xyz", 5, 5)
             )pbdoc")
        .def("set_geometry", &SteepestDescentOptimizer::set_geometry,
             py::arg("atom_data"),
             R"pbdoc(
                Set molecular geometry from list of atomic data.
                
                Args:
                    atom_data (list): List of tuples, each containing 
                                     (atomic_number, x, y, z) where coordinates 
                                     are in Bohr (atomic units).
                
                Example:
                    >>> # Two hydrogen atoms along x-axis, 1.4 Bohr apart
                    >>> optimizer.set_geometry([(1, 0.0, 0.0, 0.0), 
                    ...                         (1, 1.4, 0.0, 0.0)])
             )pbdoc")
        .def("get_geometry", &SteepestDescentOptimizer::get_geometry,
             R"pbdoc(
                Get current molecular geometry.
                
                Returns:
                    list: List of tuples (atomic_number, x, y, z) for each atom.
                         Coordinates are in Bohr (atomic units).   
             )pbdoc")
        .def("calculate_energy", &SteepestDescentOptimizer::calculate_energy,
             R"pbdoc(
                Calculate total energy at current geometry.
                
                Runs a full SCF calculation at the current molecular geometry using 
                unrestricted Hartree-Fock (UHF) with STO-3G basis sets.
                
                Returns:
                    float: Total energy in electronvolts (eV)
                
             )pbdoc")
        .def("calculate_gradient", &SteepestDescentOptimizer::calculate_gradient_numpy,
             R"pbdoc(
                Calculate energy gradient at current geometry.
                
                Returns:
                    numpy.ndarray: Gradient array with shape (3, num_atoms) in eV/Bohr.
                                  Each column corresponds to one atom, rows are x, y, z 
                                  components of the gradient.
             )pbdoc")
        .def("optimize", &SteepestDescentOptimizer::optimize,
             py::arg("gradient_tol") = 1e-4,
             py::arg("max_iterations") = 100,
             py::arg("output_path") = "",
             py::arg("logging") = false,
             R"pbdoc(
                Run steepest descent geometry optimization.
                
                Args:
                    gradient_tol (float, optional): Convergence threshold for gradient 
                                                    norm in eV/Bohr. Default: 1e-4
                    max_iterations (int, optional): Maximum number of optimization steps. 
                                                   Default: 100
                    output_path (str, optional): Path to save optimized geometry in XYZ 
                                                format. If empty, geometry is not saved.
                    logging (bool, optional): Enable verbose output including header banner
                                             and convergence summary. Default: False
                
                Example:
                    >>> optimizer.optimize(gradient_tol=5e-5, 
                    ...                   max_iterations=50,
                    ...                   output_path="optimized.xyz",
                    ...                   logging=True)
                    >>> print(f"Converged at energy: {optimizer.calculate_energy():.6f} eV")
             )pbdoc")
        .def("num_atoms", &SteepestDescentOptimizer::num_atoms,
             R"pbdoc(
                Get number of atoms in the molecule.
                
                Returns:
                    int: Number of atoms        
             )pbdoc")
        .def("save_geometry", &SteepestDescentOptimizer::save_geometry,
             py::arg("filename"),
             R"pbdoc(
                Save current geometry to XYZ file.
                
                Writes the current molecular geometry to a file in standard XYZ format.
                
                Args:
                    filename (str): Path to output XYZ file            
             )pbdoc")
        .def("__repr__", [](const SteepestDescentOptimizer &opt) {
            return "<SteepestDescentOptimizer with " + std::to_string(opt.num_atoms()) + " atoms>";
        });
}
