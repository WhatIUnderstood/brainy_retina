#pragma once
#include <vector>
#include <cmath>
#include <exception>

template <typename Floating,
          typename = std::enable_if_t<std::is_floating_point<Floating>::value>>
struct Graph
{
    std::vector<Floating> x;
    std::vector<Floating> y;
};

namespace interp_utils
{

/**
 * @brief
 *
 * @tparam Floating
 * @tparam std::enable_if_t<std::is_floating_point<Floating>::value>
 * @param x
 * @param x_values  should be sorted
 * @param y_values
 * @return Floating
 */
template <typename Floating,
          typename = std::enable_if_t<std::is_floating_point<Floating>::value>>
inline Floating lin_interp(const Floating &x, const std::vector<Floating> x_values, const std::vector<Floating> y_values, double left, double right)
{
    if (x_values.empty())
    {
        throw std::invalid_argument("lin_interp: x_values should not be empty!");
    }
    if (y_values.size() != x_values.size())
    {
        throw std::invalid_argument("lin_interp: x_values and y_values should have the same size!");
    }

    // lower_bound return greater or equal to value iterator
    typename std::vector<Floating>::const_iterator lower_it = std::lower_bound(x_values.begin(), x_values.end(), x);
    if (lower_it == x_values.begin())
    {
        if (x == x_values.front())
        {
            return y_values.front();
        }
        else
        {
            return left;
        }
    }
    else if (lower_it == x_values.end())
    {
        if (x == x_values.back())
        {
            return y_values.back();
        }
        else
        {
            return right;
        }
    }
    else
    {
        //Make the linear interpolation

        // the index retreival is O(n) but as I use this function only at initialisation I prefer to keep the code cleaner
        auto second_index = lower_it - x_values.begin();
        auto first_index = second_index - 1;

        const double steep = (y_values[second_index] - y_values[first_index]) / (x_values[second_index] - x_values[first_index]);
        return y_values[first_index] + steep * (x - x_values[first_index]);
    }
}

template <typename Floating,
          typename = std::enable_if_t<std::is_floating_point<Floating>::value>>
inline Floating lin_interp(const Floating &x, const std::vector<Floating> x_values, const std::vector<Floating> y_values)
{
    lin_interp(x, x_values, y_values, x_values.front(), x_values.back());
}

template <typename Floating,
          typename = std::enable_if_t<std::is_floating_point<Floating>::value>>
inline std::vector<Floating> lin_interp_integral(const std::vector<Floating> x_values, const std::vector<Floating> y_values)
{
    typename std::vector<Floating> integral;
    integral.push_back(0);
    for (unsigned int i = 1; i < y_values.size(); i++)
    {
        if (y_values[i] < 0 || y_values[i - 1] < 0)
        {
            throw std::invalid_argument("lin_interp_integral: negative values not supported");
        }
        const Floating dy = std::abs(y_values[i] - y_values[i - 1]);
        const Floating dx = x_values[i] - x_values[i - 1];
        integral.push_back(integral[i - 1] + dx * dy / 2.0 + dx * std::min(y_values[i], y_values[i - 1]));
    }
    return integral;
}

template <typename FUNCTION, typename Floating,
          typename = std::enable_if_t<std::is_floating_point<Floating>::value>>
inline Graph<Floating> buildGraph(FUNCTION function_callback, Floating from_x, Floating to_x, Floating step_x)
{
    Graph<Floating> graph;
    Floating current_x = from_x;
    while (current_x <= to_x)
    {
        graph.x.push_back(current_x);
        graph.y.push_back(function_callback(current_x));
        current_x += step_x;
    }

    return graph;
}

template <typename FUNCTION, typename Floating,
          typename = std::enable_if_t<std::is_floating_point<Floating>::value>>
Graph<Floating> computeTransformedIntegral(FUNCTION transform_function, const Graph<Floating> &graph)
{
    std::vector<double> transformed_y;
    for (const auto &value : graph.y)
    {
        transformed_y.push_back(transform_function(value));
    }

    Graph<Floating> result;
    result.y = interp_utils::lin_interp_integral(graph.x, transformed_y);
    result.x = graph.x;
    return result;
}

///////////// Retina specialisation ///////////////////////////

template <typename FUNCTION, typename Floating,
          typename = std::enable_if_t<std::is_floating_point<Floating>::value>>
Graph<Floating> computeLinearDensityIntegral(const Graph<Floating> &density_graph)
{
    return computeTransformedIntegral([](const Floating &val) { return std::sqrt(val); }, density_graph);
}

} // namespace interp_utils