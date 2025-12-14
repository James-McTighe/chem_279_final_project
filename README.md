# Chem 279 Final Project - Steepest Descent Optimizer Feature Branch

## Optimizer Instructions

1. Open Dev Container in VS code or build image with bash build_image.sh
2. run interactive session with bash interactive.sh if not utilizing Dev Container.
3. Compile program with bash build.sh inside container
4. Run executable with ./build/steepest_descent MolJSON/H2.json (Or other molecule).
5. Outputs are printed to terminal and saved in xyz format in the output directory.


### Building the Python Module

1. **Build the optimizer module:**
   ```bash
   ./build_python_module.sh
   ```
   This compiles the `steepest_descent_py` module and places it in the `build/` directory. or:

   ```bash
   ./build_python_module.sh --install
   ```
   This also copies the module to your Python site-packages for system-wide access.

### Using the Optimizer

#### Python Script Usage

```python
import steepest_descent_py as sd

# Initialize optimizer from a molecular JSON file
optimizer = sd.SteepestDescentOptimizer(
    atoms_file_path="MolJSON/H2.json",
    num_alpha_electrons=1,
    num_beta_electrons=1
)


#### Key Methods

- `calculate_energy()` - Compute the electronic energy at current geometry
- `calculate_gradient()` - Compute energy gradient with respect to atomic positions
- `optimize(gradient_tol, max_iterations, output_path)` - Run steepest descent optimization
- `get_geometry()` - Retrieve current atomic positions as list of (Z, x, y, z) tuples
- `set_geometry(geom)` - Set atomic positions from list of (Z, x, y, z) tuples
- `num_atoms()` - Get number of atoms in the system
- `save_geometry(path)` - Save current geometry to XYZ file

Sources/Credit:
- Chemx79 image as skeleton for our docker image: https://github.com/Berkeley-Chem-179-279/hw-common-utils