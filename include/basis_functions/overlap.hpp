#pragma once
#include <armadillo>
#include <vector>

#include "basis.hpp"
#include "gaussian.hpp"
#include "math.hpp"

// from hw4 solution

struct Shell_Info
{
    arma::vec3 center = arma::fill::zeros;
    double alpha = 0;
    int shell = 0;
};

inline std::vector<Gaussian> get_gaussians_in_shell(Shell_Info si)
{
    std::vector<Gaussian> v;
    for ( int i = si.shell; i >= 0; i-- )
    {
        for ( int j = si.shell; j >= 0; j-- )
        {
            for ( int k = si.shell; k >= 0; k-- )
            {
                if ( (i + j + k) == si.shell )
                {
                    v.push_back(Gaussian(si.center, si.alpha,
                                         arma::vec3{static_cast<double>(i),
                                                    static_cast<double>(j),
                                                    static_cast<double>(k)}));
                }
            }
        }
    }
    return v;
};

inline double summation_nonsense(const Gaussian g1, const Gaussian g2, arma::vec3 Rp,
                          int dim)
{
    int la = g1.power[dim];
    int lb = g2.power[dim];
    double sum = 0;
    for ( int i = 0; i <= la; i++ )
    {
        for ( int j = 0; j <= lb; j++ )
        {
            if ( (i + j) % 2 == 0 )
            {
                double element = 1;
                element *= binomial(la, i);
                element *= binomial(lb, j);
                element *= double_factorial(i + j - 1);
                element *= std::pow(Rp[dim] - g1.center[dim], la - i);
                element *= std::pow(Rp[dim] - g2.center[dim], lb - j);
                element /= std::pow(2 * (g1.alpha + g2.alpha), (i + j) / 2);
                sum += element;
            }
        }
    }
    return sum;
}

inline double analytical_S_ab_dim(const Gaussian g1, const Gaussian g2, int dim)
{
    double gamma = g1.alpha + g2.alpha;
    double result = 1;
    arma::vec3 Rp = (g1.alpha * g1.center + g2.alpha * g2.center) / (gamma);
    result *= std::exp(-g1.alpha * g2.alpha *
                       std::pow(g1.center[dim] - g2.center[dim], 2) / (gamma));
    result *= std::sqrt(M_PI / (gamma));
    result *= summation_nonsense(g1, g2, Rp, dim);
    return result;
}

inline double analytical_gaussian_overlap(const Gaussian g1, const Gaussian g2)
{
    double result = 1;
    for ( int i = 0; i < 3; i++ )
    {
        result *= analytical_S_ab_dim(g1, g2, i);
    }
    return result;
}

inline arma::mat calculate_overlap_matrix(std::vector<Gaussian> vg1,
                                   std::vector<Gaussian> vg2)
{
    arma::mat m = arma::mat(vg1.size(), vg2.size());
    for ( int i = 0; i < vg1.size(); i++ )
    {
        for ( int j = 0; j < vg2.size(); j++ )
        {
            m(i, j) = analytical_gaussian_overlap(vg1.at(i), vg2.at(j));
        }
    }
    return m;
}
