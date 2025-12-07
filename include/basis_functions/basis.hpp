#pragma once

#include "atom.hpp"
#include "gaussian.hpp"
#include "overlap.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

// from hw4 solution

namespace fs = std::filesystem;
using json = nlohmann::json;

struct ContractedGaussian
{
    arma::vec3 center{};
    std::array<Gaussian, 3> prim{};
    arma::vec3 power{};       // stores primitive exponents α_k
    arma::vec3 contraction{}; // stores contraction coeffs c_k
    arma::vec3 norms{};       // primitive norms N_k
    arma::vec3 momentum{};    // (lx, ly, lz)
    fs::path file_path;

    int z_star;
    double ion_term;
    double beta;
    int atom_index;

    std::string atom = "X";
    std::string type = "Base";

    // If other code depends on these names, keep them in sync:
    Gaussian prim1, prim2, prim3;

    explicit ContractedGaussian(arma::vec3 center) : center(center) {}

    void set_parameters()
    {
        // make sure JSON looks good
        if ( file_path.empty() )
            throw std::runtime_error("ContractedGaussian::file_path is empty.");

        std::ifstream file(file_path);
        if ( !file )
            throw std::runtime_error("Could not open JSON file: " +
                                     file_path.string());

        json config;
        file >> config; // RAII closes automatically

        // Validate and read array
        if ( !config.contains("contracted_gaussians") ||
             !config["contracted_gaussians"].is_array() )
            throw std::runtime_error(
                "JSON must contain array 'contracted_gaussians'.");

        const auto & arr = config["contracted_gaussians"];
        if ( arr.empty() )
            throw std::runtime_error("'contracted_gaussians' is empty.");

        // assign parameters read from JSON file
        for ( int k = 0; k < 3; ++k )
        {
            // some additional error validation
            const auto & item = arr.at(k);
            if ( !item.contains("exponent") ||
                 !item.contains("contraction_coefficient") )
                throw std::runtime_error("Each primitive must have 'exponent' "
                                         "and 'contraction_coefficient'.");

            double exp = item.at("exponent").get<double>();
            double coeff = item.at("contraction_coefficient").get<double>();
            if ( exp <= 0.0 )
                throw std::runtime_error(
                    "Primitive exponent must be > 0."); // edge case just in
                                                        // case I overwrite a
                                                        // value by accident

            // actual parameter assignemnt
            power[k] = exp;
            contraction[k] = coeff;
        }

        // Optional: set metadata if present
        if ( config.contains("atomic_number") )
        {
            atom =
                std::to_string(config["atomic_number"]
                                   .get<int>()); // or map to symbol if you like
        }
        if ( config.contains("shell_momentum") )
        {
            // you can store/validate this if helpful
        }
    }

    void build_primitives_and_norms()
    {
        // Initialize primitive Gaussians
        for ( int k = 0; k < 3; ++k )
        {
            prim[k] = Gaussian(center, power[k], momentum);
        }

        // Keep mirrors
        prim1 = prim[0];
        prim2 = prim[1];
        prim3 = prim[2];

        // Compute normalization constants
        for ( int k = 0; k < 3; ++k )
        {
            if ( power[k] <= 0.0 )
            {
                norms[k] = 0.0;
                continue;
            } // unused slot
            double Skk = analytical_gaussian_overlap(prim[k], prim[k]);
            norms[k] = 1.0 / std::sqrt(Skk);
        }
    }
};

/******************
    Carbon Contracted Gaussians

 */

struct Cs : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Cs(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 0.0, 0.0};
        atom = "C";
        type = "S";
        z_star = 4;
        ion_term = 14.051;
        beta = -21;

        file_path = fs::path("basis/C_s_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Cpx : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Cpx(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{1.0, 0.0, 0.0};
        atom = "C";
        type = "P";
        z_star = 4;
        ion_term = 5.572;
        beta = -21;

        file_path = fs::path("basis/C_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Cpy : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Cpy(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 1.0, 0.0};
        atom = "C";
        type = "P";
        z_star = 4;
        ion_term = 5.572;
        beta = -21;

        file_path = fs::path("basis/C_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Cpz : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Cpz(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 0.0, 1.0};
        atom = "C";
        type = "P";
        z_star = 4;
        ion_term = 5.572;
        beta = -21;

        file_path = fs::path("basis/C_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

/******************
    Hydrogen Contracted Gaussians

 */

struct Hs : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Hs(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 0.0, 0.0};
        atom = "H";
        type = "S";
        z_star = 1;
        ion_term = 7.176;
        beta = -9;

        file_path = fs::path("basis/H_s_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

/******************
    Fluorine Contracted Gaussians

 */

struct Fs : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Fs(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 0.0, 0.0};
        atom = "F";
        type = "S";
        z_star = 7;
        ion_term = 32.272;
        beta = -39;

        file_path = fs::path("basis/F_s_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Fpx : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Fpx(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{1.0, 0.0, 0.0};
        atom = "F";
        type = "P";
        z_star = 7;
        ion_term = 11.080;
        beta = -39;

        file_path = fs::path("basis/F_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};
struct Fpy : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Fpy(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 1.0, 0.0};
        atom = "F";
        type = "P";
        z_star = 7;
        ion_term = 11.080;
        beta = -39;

        file_path = fs::path("basis/F_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};
struct Fpz : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Fpz(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 0.0, 1.0};
        atom = "F";
        type = "P";
        z_star = 7;
        ion_term = 11.080;
        beta = -39;

        file_path = fs::path("basis/F_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

/******************
    Nitrogen Contracted Gaussians

 */

struct Ns : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Ns(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 0.0, 0.0};
        atom = "N";
        type = "S";
        z_star = 5;
        ion_term = 19.316;
        beta = -25;

        file_path = fs::path("basis/N_s_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Npx : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Npx(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{1.0, 0.0, 0.0};
        atom = "N";
        type = "P";
        z_star = 5;
        ion_term = 7.275;
        beta = -25;

        file_path = fs::path("basis/N_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Npy : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Npy(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 1.0, 0.0};
        atom = "N";
        type = "P";
        z_star = 5;
        ion_term = 7.275;
        beta = -25;

        file_path = fs::path("basis/N_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Npz : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Npz(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 0.0, 1.0};
        atom = "N";
        type = "P";
        z_star = 5;
        ion_term = 7.275;
        beta = -25;

        file_path = fs::path("basis/N_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

/******************
    Oxygen Contracted Gaussians

 */

struct Os : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Os(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 0.0, 0.0};
        atom = "O";
        type = "S";
        z_star = 6;
        ion_term = 25.390;
        beta = -31;

        file_path = fs::path("basis/O_s_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Opx : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Opx(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{1.0, 0.0, 0.0};
        atom = "O";
        type = "P";
        z_star = 6;
        ion_term = 9.111;
        beta = -31;

        file_path = fs::path("basis/O_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Opy : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Opy(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 1.0, 0.0};
        atom = "O";
        type = "P";
        z_star = 6;
        ion_term = 9.111;
        beta = -31;

        file_path = fs::path("basis/O_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

struct Opz : ContractedGaussian
{
    using ContractedGaussian::ContractedGaussian;

    explicit Opz(arma::vec3 center) : ContractedGaussian(center)
    {
        momentum = arma::vec3{0.0, 0.0, 1.0};
        atom = "O";
        type = "P";
        z_star = 6;
        ion_term = 9.111;
        beta = -31;

        file_path = fs::path("basis/O_p_STO3G.json");
        set_parameters();
        build_primitives_and_norms();
    }
};

// Functions

inline double contracted_overlap(const ContractedGaussian A,
                                 const ContractedGaussian B)
{
    double S = 0.0;
    for ( int k = 0; k < 3; ++k )
    {
        for ( int l = 0; l < 3; ++l )
        {
            double Skl = analytical_gaussian_overlap(A.prim[k], B.prim[l]);
            S += A.contraction[k] * B.contraction[l] * A.norms[k] * B.norms[l] *
                 Skl;
        }
    }
    return S;
}

inline arma::mat
build_overlap_matrix(const std::vector<ContractedGaussian> & basis)
{
    const int N = static_cast<int>(basis.size());
    arma::mat S(N, N, arma::fill::zeros);
    for ( int i = 0; i < N; ++i )
    {
        for ( int j = 0; j <= i; ++j )
        {
            double sij = contracted_overlap(basis[i], basis[j]);
            S(i, j) = S(j, i) = sij;
        }
    }
    return S;
}

// build basis set
inline std::vector<ContractedGaussian>
make_sto3g_basis_from_xyz(std::vector<Atom> & atoms)
{
    std::vector<ContractedGaussian> basis;

    for ( int i = 0; i < atoms.size(); ++i )
    {
        int Z = atoms[i].z_num;
        arma::vec3 R = atoms[i].pos;

        // lambda function to add the atom index to each basis set
        auto tag = [i](ContractedGaussian cg) {
            cg.atom_index = i;
            return cg;
        };

        switch ( Z )
        {
        case 1: // Hydrogen: 1s
            basis.push_back(tag(Hs(R)));
            break;

        case 6: // Carbon: 2s, 2px, 2py, 2pz
            basis.push_back(tag(Cs(R)));
            basis.push_back(tag(Cpx(R)));
            basis.push_back(tag(Cpy(R)));
            basis.push_back(tag(Cpz(R)));
            break;

        case 7: // Nitrogen: 2s, 2px, 2py, 2pz
            basis.push_back(tag(Ns(R)));
            basis.push_back(tag(Npx(R)));
            basis.push_back(tag(Npy(R)));
            basis.push_back(tag(Npz(R)));
            break;

        case 8: // Oxygen: 2s, 2px, 2py, 2pz
            basis.push_back(tag(Os(R)));
            basis.push_back(tag(Opx(R)));
            basis.push_back(tag(Opy(R)));
            basis.push_back(tag(Opz(R)));
            break;

        case 9: // Fluorine: 2s, 2px, 2py, 2pz
            basis.push_back(tag(Fs(R)));
            basis.push_back(tag(Fpx(R)));
            basis.push_back(tag(Fpy(R)));
            basis.push_back(tag(Fpz(R)));
            break;

        default:
            throw std::runtime_error("Unsupported Z in STO-3G builder: " +
                                     std::to_string(Z));
        }
    }
    return basis;
}