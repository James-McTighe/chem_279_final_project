#pragma once

#include <algorithm>
#include <armadillo>

// from hw4 solution
// used file parsing from hw3 because I liked it better

struct Atom
{
    int z_num;
    int vector_idx;
    arma::vec3 pos;
    int z_star;
    double ion_term_s;
    double ion_term_p;
    double beta;
    double P_total_alpha;
    double P_total_beta;
    double mass;
};

inline double nuclear_repulsion(std::vector<Atom> atoms)
{
    double energy = 0.0;

    int N = atoms.size();

    for ( int i = 0; i < N; ++i )
        for ( int j = 0; j < i; ++j )
        {
            Atom A = atoms[i];
            Atom B = atoms[j];

            double R = arma::norm(A.pos - B.pos);

            energy += A.z_star * B.z_star / R;
        }

    return energy;
}


inline arma::vec3 nuclear_repulsion_derivitive_atom_specific(const Atom & target,
                                                            const std::vector<Atom> & atoms)
{
    arma::vec3 repulsion_gradient(arma::fill::zeros);

    for ( const Atom& atom : atoms )
    {
        if ( atom.vector_idx == target.vector_idx )
            continue;

        arma::vec3 Rij = target.pos - atom.pos;
        double r_mag = arma::norm(Rij);
        double r3 = std::pow(r_mag, 3);

        // Standard physics: Force on A due to B = (Za*Zb / r^3) * (Ra - Rb)
        // Energy Gradient is the NEGATIVE of the force: dE/dRa = -Force
        arma::vec3 term = - (target.z_star * atom.z_star * Rij) / r3;

        repulsion_gradient += term;
    }

    
    return repulsion_gradient;
}

inline arma::mat nuclear_repulsion_gradient(const std::vector<Atom> & atoms)
{
    // helper lambda
    auto gradient_helper = [atoms](Atom target) {
        return nuclear_repulsion_derivitive_atom_specific(target, atoms);
    };

    const int N = atoms.size();

    arma::mat nuclear_gradient(3, N);

    for ( int i = 0; i < N; ++i )
    {
        arma::vec3 atom_gradient = gradient_helper(atoms[i]);
        nuclear_gradient.col(i) = atom_gradient;
    }

    return nuclear_gradient;
}

inline std::vector<Atom> parse_file(std::string filepath, bool verbose = false)
{
    if ( verbose )
        std::cout << "Attempting to parse: " << filepath << std::endl;

    std::vector<Atom> atoms;

    std::ifstream file(filepath);

    if ( !file.is_open() )
    {
        throw std::runtime_error("File could not be opened");
    }

    std::string line;
    std::getline(file,
                 line); // remove first line, which has the number of atoms
    std::getline(file,
                 line); // remove second line, which is a comment (standardizes to MP xyz format)

    int idx = 0; // keep track of which atom is which so a single one can be
                 // identified relative to the whole list

    while ( std::getline(file, line) )
    {
        std::istringstream linestream(line);

        Atom a;
        arma::vec3 p;

        // Try to read as atomic number first, if that fails, try as element symbol (To work with HW5 xyz format and MP xyz format)
        std::string first_token;
        linestream >> first_token;
        
        // Check if it's a number or element symbol
        if ( std::isdigit(first_token[0]) )
        {
            // It's an atomic number
            a.z_num = std::stoi(first_token);
        }
        else
        {
            // It's an element symbol, convert to atomic number
            if ( first_token == "H" ) a.z_num = 1;
            else if ( first_token == "C" ) a.z_num = 6;
            else if ( first_token == "O" ) a.z_num = 8;
            else if ( first_token == "F" ) a.z_num = 9;
            else
            {
                throw std::runtime_error("Unknown element symbol: " + first_token);
            }
        }
        
        linestream >> p[0];
        linestream >> p[1];
        linestream >> p[2];

        if ( verbose )
            std::cout << a.z_num << "(" << p[0] << ", " << p[1] << ", " << p[2]
                      << ")" << std::endl;

        a.pos = p;
        a.vector_idx = idx;

        switch ( a.z_num )
        {
        case 1:
            a.z_star = 1;
            a.ion_term_s = 7.176;
            a.beta = -9;
            a.mass = 1.0078;
            break;
        case 6:
            a.z_star = 4;
            a.ion_term_s = 14.051;
            a.ion_term_p = 5.572;
            a.beta = -21;
            a.mass = 12.011;
            break;
        case 7:
            a.z_star = 5;
            a.ion_term_s = 19.316;
            a.ion_term_p = 7.275;
            a.beta = -25;
            a.mass = 14.007;
            break;
        case 8:
            a.z_star = 6;
            a.ion_term_s = 25.390;
            a.ion_term_p = 9.111;
            a.beta = -31;
            a.mass = 15.999;
            break;
        case 9:
            a.z_star = 7;
            a.ion_term_s = 32.272;
            a.ion_term_p = 11.080;
            a.beta = -39;
            a.mass = 18.998;
            break;

        default:
            break;
        }
        idx++;

        // This is cheating but I would use map.contains if I was allowed to use
        // c++ 20

        std::vector<int> allowed_elements = {1, 6, 7, 8, 9};

        auto it = std::find(allowed_elements.begin(), allowed_elements.end(),
                            a.z_num);

        if ( it == allowed_elements.end() )
        {
            throw std::runtime_error(
                "One of the elements in the input was not as expected");
        } else
        {
            atoms.push_back(a);
        }
    }

    file.close();

    return atoms;
}