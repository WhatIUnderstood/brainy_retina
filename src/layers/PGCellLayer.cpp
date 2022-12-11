#include "PGCellLayer.h"

#include <iostream>
#include <map>

#include "utils/Random.h"
#include "utils/polar_utils.h"

#ifdef WITH_MATPLOTLIB
#include "matplotlib_cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

void PGCellLayer::plotGraphs() {
#ifdef WITH_MATPLOTLIB
  plt::figure();
  plt::named_plot("radius 1/8th", description_map_["gcp_radius_at_eighth_x"], description_map_["gcp_radius_at_eighth"]);
  // plt::named_plot("radius newt 1/8th", description_map_["gcp_radius_next_at_eighth_x"],
  // description_map_["gcp_radius_next_at_eighth"]);
  plt::title("Test");
  plt::legend();

  plt::figure();
  plt::named_plot("GCp angular pos 1/8th", description_map_["gc_parasol_angular_eighth_pose_x"],
                  description_map_["gc_parasol_angular_eighth_pose"]);
  plt::named_plot("GCp angular pos 1/2th", description_map_["gc_parasol_angular_half_pose_x"],
                  description_map_["gc_parasol_angular_half_pose"]);
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
  plt::named_plot("GCp cone center 1/8th", description_map_["gc_parasol_cone_eighth_pose_x"],
                  description_map_["gc_parasol_cone_eighth_pose"]);
  plt::named_plot("GCp cone center 1/2th", description_map_["gc_parasol_cone_half_pose_x"],
                  description_map_["gc_parasol_cone_half_pose"]);
  plt::title("Ganglionar cells cone center");
  plt::legend();

  plt::figure();
  plt::named_plot("GCp cone center 1/8th", description_map_["gc_parasol_cone_eighth_pose_deg"],
                  description_map_["gc_parasol_cone_eighth_pose"]);
  plt::named_plot("GCp cone center 1/2th", description_map_["gc_parasol_cone_half_pose_deg"],
                  description_map_["gc_parasol_cone_half_pose"]);
  plt::title("Ganglionar cells cone center (deg)");
  plt::legend();

  plt::figure();
  plt::named_plot("GCp density 1/2th", description_map_["gcp_density_deg"], description_map_["gcp_density"]);
  plt::named_plot("Cone density at 1/2th", description_map_["gcp_density_deg"], description_map_["cone_density"]);
  plt::title("GCp vs Cone densities (deg)");
  plt::legend();

  plt::figure();
  plt::named_plot("GCp cone ratio 1/2th", description_map_["gcp_density_deg"], description_map_["gcp_cone_ratio"]);
  plt::title("GCp / Cone ratio (deg)");
  plt::legend();

  plt::figure();
  plt::named_plot("GCp cone ratio 1/2th", description_map_["pgc_cones_in_receptive_at_half_deg"],
                  description_map_["pgc_cones_in_receptive_at_half"]);
  plt::title("Cones in parasol receptive field");
  plt::legend();

#endif
}