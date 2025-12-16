# chem_279_final_project

## Instructions

1. Build the docker image using the `build_image.sh` script.
2. Start an interactive session in the built container with bash `interactive.sh`
3. Compile the source code using `build.sh`
4. To perform calculations on all molcule files located in the `MolJSON` directory, run `run.sh`
    - Alternatively, if only a single molecule is the target of analysis, simply run `run.sh` followed by the file path of the molecule JSON file.

5. Outputs are are saved under the `output` directory in `.out` file form.

Sources/Credit:
- Chemx79 image as skeleton for our docker image:
[Docker Base Image link](https://github.com/Berkeley-Chem-179-279/hw-common-utils)
- Optimize Molecule geometries from [Materials Project](https://next-gen.materialsproject.org/)

