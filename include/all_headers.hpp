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

#include "gaussian.hpp"
#include "atom.hpp"
#include "basis.hpp"
#include "fock.hpp"
#include "gradient.hpp"
#include "hamiltonian.hpp"
#include "math.hpp"
#include "overlap.hpp"

#include "golden.hpp"
