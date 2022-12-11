#include "MGCellLayer.h"

#include <iostream>
#include <map>

#include "utils/Random.h"
#include "utils/polar_utils.h"

#ifdef WITH_MATPLOTLIB
#include "matplotlib_cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

void MGCellLayer::plotGraphs() {
#ifdef WITH_MATPLOTLIB
  plt::figure();
  plt::named_plot("radius 1/8th", description_map_["gcm_radius_at_eighth_x"], description_map_["gcm_radius_at_eighth"]);
  // plt::named_plot("radius newt 1/8th", description_map_["gcm_radius_next_at_eighth_x"],
  // description_map_["gcm_radius_next_at_eighth"]);
  plt::title("Test");
  plt::legend();

  plt::figure();
  plt::named_plot("GCm angular pos 1/8th", description_map_["gc_midget_angular_eighth_pose_x"],
                  description_map_["gc_midget_angular_eighth_pose"]);
  plt::named_plot("GCm angular pos 1/2th", description_map_["gc_midget_angular_half_pose_x"],
                  description_map_["gc_midget_angular_half_pose"]);
  plt::title("Ganglionar cells angular pose");
  plt::legend();
  //
  plt::figure();
  plt::named_plot("GC ext radius 1/8th", description_map_["gc_external_radius_at_eighth_x"],
                  description_map_["gc_external_radius_at_eighth"]);
  plt::named_plot("GC ext radius 1/2th", description_map_["gc_external_radius_at_half_x"],
                  description_map_["gc_external_radius_at_half"]);
  plt::title("Ganglionar cells radius");
  plt::legend();

  plt::figure();
  plt::named_plot("GC ext radius 1/2th", description_map_["gc_external_radius_at_half_deg"],
                  description_map_["gc_external_radius_at_half"]);
  plt::title("Ganglionar cells radius (deg)");
  plt::xlabel("Eccentricity (degrees)");
  plt::ylabel("Radius in cone diameter unit");
  plt::legend();

  plt::figure();
  plt::named_plot("GCm cone center 1/8th", description_map_["gc_midget_cone_eighth_pose_x"],
                  description_map_["gc_midget_cone_eighth_pose"]);
  plt::named_plot("GCm cone center 1/2th", description_map_["gc_midget_cone_half_pose_x"],
                  description_map_["gc_midget_cone_half_pose"]);
  plt::title("Ganglionar cells cone center");
  plt::legend();

  plt::figure();
  plt::named_plot("GCm cone center 1/8th", description_map_["gc_midget_cone_eighth_pose_deg"],
                  description_map_["gc_midget_cone_eighth_pose"]);
  plt::named_plot("GCm cone center 1/2th", description_map_["gc_midget_cone_half_pose_deg"],
                  description_map_["gc_midget_cone_half_pose"]);
  plt::title("Ganglionar cells cone center (deg)");
  plt::legend();

  plt::figure();
  plt::named_plot("GCm density 1/2th", description_map_["gcm_density_deg"], description_map_["gcm_density"]);
  plt::named_plot("Cone density at 1/2th", description_map_["gcm_density_deg"], description_map_["cone_density"]);
  plt::title("GCm vs Cone densities (deg)");
  plt::legend();

  plt::figure();
  plt::named_plot("GCm cone ratio 1/2th", description_map_["gcm_density_deg"], description_map_["gcm_cone_ratio"]);
  plt::title("GCm / Cone ratio (deg)");
  plt::legend();
#endif
}