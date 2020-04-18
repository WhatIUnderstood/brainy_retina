#include "ConeLayer.h"
#include <iostream>
#include <map>

#include "Utils/Random.h"
#include "Utils/polar_utils.h"

#ifdef WITH_MATPLOTLIB
#include "matplotlib_cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

void ConeLayer::plotGraphs()
{
#ifdef WITH_MATPLOTLIB
    plt::figure();
    plt::named_plot("Cone pixel index 1/8th", description_map_["pixel_index_x"], description_map_["pixel_index"]);
    plt::title("Cone pixel index");
    plt::legend();
    //
    plt::figure();
    plt::named_plot("Cone index 1/2th", description_map_["cone_index_at_half_x"], description_map_["cone_index_at_half"]);
    plt::title("Cone index");
    plt::legend();

    plt::figure();
    plt::named_plot("Cone index 1/2th", description_map_["cone_index_at_half_deg"], description_map_["cone_index_at_half"]);
    plt::title("Cone index (deg)");
    plt::legend();

#endif
}