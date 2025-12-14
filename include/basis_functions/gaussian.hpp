#pragma once

#include <armadillo>
#include <cmath>
#include <functional>
#include <stdexcept>


// taken from hw2 solution
class Gaussian
{
public:
    arma::vec3 center;
    double alpha;
    arma::vec3 power;
    Gaussian() {};
    Gaussian(arma::vec3 center, double alpha, arma::vec3 power)
        : center(center), alpha(alpha), power(power) {};

    // define a nice function to evaluate the value at a position
    double operator()(const arma::vec3 p) const
    {
        return std::pow((p[0] - center[0]), power[0]) *
               std::pow((p[1] - center[1]), power[1]) *
               std::pow((p[2] - center[2]), power[2]) *
               std::exp(-alpha * std::pow(arma::norm(p - center), 2));
    }

    // this makes print debuging easier
    friend std::ostream &operator<<(std::ostream &os, const Gaussian &g)
    {
        os << "center: (" << g.center[0] << ", " << g.center[1] << ", "
           << g.center[2] << ") alpha: " << g.alpha << " power: (" << g.power[0]
           << ", " << g.power[1] << ", " << g.power[2] << ")";
        return os;
    }
};
