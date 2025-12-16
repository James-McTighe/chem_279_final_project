# Chem 279 Final Project - Steepest Descent Molecular Geometry Optimization and IR Spectroscopy Computation

## IR Spec Instructions

1. Open Dev Container in VS code or build image with bash build_image.sh
2. run interactive session with bash interactive.sh if not utilizing Dev Container.
3. Compile program with bash build.sh inside container
4. Run executable with ./build/steepest_descent MolJSON/H2.json (Or other molecule).
5. Outputs are printed to terminal and saved in xyz format in the output directory.
5. Outputs are are saved under the `output` directory in `.out` file form.
6. OPTIONAL: User may elect to use .xyz file directly with --xyz option. If option is selected the num_alpha_elec and num_beta elec must be passed consecutively after .xyz path.
    Use Example: 
   * `./build/ir_spec include/atoms/H2.xyz 1 1`

## Optimizer Instructions

1. Build the docker image using the `build_image.sh` script.
2. Start an interactive session in the built container with bash `interactive.sh`
3. Compile the source code using `build.sh`
4. To perform calculations on all molcule files located in the `MolJSON` directory, run `run.sh`
    - Alternatively, if only a single molecule is the target of analysis, simply run `run.sh` followed by the file path of the molecule JSON file.
    
### Using the Optimizer

#### Python Script Usage
* Ensure python module in the same directory as script, or add to python environment if using with Jupyter notebook.

```python
import steepest_descent_py as sd

# Initialize optimizer from a molecular JSON file
optimizer = sd.SteepestDescentOptimizer(
    atoms_file_path="MolJSON/H2.json",
    num_alpha_electrons=1,
    num_beta_electrons=1
)

#### Key Methods

- calculate_energy() - Compute the electronic energy at
                        current geometry

- calculate_gradient() Compute energy gradient with respect
                        to atomic positions

- optimize(gradient_tol, max_iterations, output_path) 
                        - Run steepest descent optimization

- get_geometry() - Retrieve current atomic positions as list
                    of (Z, x, y, z) tuples

- set_geometry(geom) - Set atomic positions from list of 
                        (Z, x, y, z) tuples

- num_atoms() - Get number of atoms in the system

- save_geometry(path) - Save current geometry to XYZ file.

```
Sources/Credit:
- Chemx79 image as skeleton for our docker image:
[Docker Base Image link](https://github.com/Berkeley-Chem-179-279/hw-common-utils)
- Optimize Molecule geometries from [Materials Project](https://next-gen.materialsproject.org/)
