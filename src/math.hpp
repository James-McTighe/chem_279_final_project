#pragma once

// from hw4 solution

int double_factorial(int i)
{
    int result = 1;
    while (i >= 1)
    {
        result *= i;
        i -= 2;
    }
    return result;
}

int factorial(int i)
{
    int result = 1;
    while (i >= 1)
    {
        result *= i;
        i -= 1;
    }
    return result;
}

double binomial(int m, int n)
{
    return static_cast<double>(factorial(m)) /
           static_cast<double>(factorial(n) * factorial(m - n));
}