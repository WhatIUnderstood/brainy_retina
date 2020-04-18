#pragma once

#include <vector>
#include <random>
#include <algorithm>

class Random
{

public:
    Random(int seed = 1)
    {
        mt_rand.seed(seed);
    }

    double getRandom()
    {
        //std::mt19937 mt_rand(0);//Use always the same seed for regeneration
        return mt_rand() / ((double)(std::mt19937::max()));
    }

    template <typename OUTPUT_TYPE>
    OUTPUT_TYPE weightedRandom(const std::vector<std::pair<float, OUTPUT_TYPE>> &probabilities_)
    {
        std::vector<std::pair<float, OUTPUT_TYPE>> probabilities = probabilities_;
        float proba_sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0, [](float sum, const std::pair<float, OUTPUT_TYPE> &pair) {
            return sum + pair.first;
        });
        if (proba_sum > 1)
        {
            throw std::invalid_argument("weightedRandom cannot have probabilities sum above 1");
        }
        // sort the probabilities
        std::sort(probabilities.begin(), probabilities.end(), [](const std::pair<float, OUTPUT_TYPE> &a, const std::pair<float, OUTPUT_TYPE> &b) {
            return a.first < b.first;
        });

        float random = getRandom();
        for (std::pair<float, OUTPUT_TYPE> proba : probabilities)
        {
            if (random < proba.first)
            {
                return proba.second;
            }
            else
            {
                random -= proba.first;
            }
        }

        throw std::runtime_error("weightedRandom: unexpected error");
    }

private:
    std::mt19937 mt_rand;
};