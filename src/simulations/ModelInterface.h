#pragma once

class ModelInterface
{
public:
    virtual double getMaxEccentricity() const = 0;

    virtual double getDensityAt(double ecc_deg) const = 0;

    virtual double getIndexAt(double ecc_deg) const = 0;

    virtual double getMaxIndex() const = 0;

    virtual double getEccentricityAt(double index) const = 0;
};