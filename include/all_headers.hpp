#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <armadillo>
#include <nlohmann/json.hpp>

#include "basis_functions/Gaussian.hpp"
#include "basis_functions/atom.hpp"
#include "basis_functions/basis.hpp"
#include "fock/fock.hpp"
#include "gradient/gradient.hpp"
#include "fock/hamiltonian.hpp"
#include "math.hpp"
#include "basis_functions/overlap.hpp"

#include "geometry_optimization/golden.hpp"
