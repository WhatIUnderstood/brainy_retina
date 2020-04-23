#pragma once

class DensityFunction
{
public:
    virtual double at(double ecc) const = 0;
};

