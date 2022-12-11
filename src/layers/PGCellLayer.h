#pragma once

#include <iostream>
#include <map>
#include <memory>

#include "layers/ConeLayer.h"
#include "simulations/ConeModel.h"
#include "simulations/PGCellsModel.h"
#include "utils/Random.h"
#include "utils/polar_utils.h"

/**
 * @brief Parvo Ganglionar cells layer
 *
 */
class PGCellLayer {
 public:
  PGCellLayer(std::unique_ptr<PGCellsModel> &pgc_model_ptr, const ConeLayer &cone_layer, int seed = 1)
      : pgc_model_ptr_(std::move(pgc_model_ptr)), random(seed) {
    const Cones &cones_cpu = cone_layer.cones();
    const ConeModel &cone_model = cone_layer.coneModel();

    std::cout << "parasol cells model radius: " << pgc_model_ptr_->getMaxIndex() << " cells" << std::endl;
    std::cout << "parasol cells model max eccentricity: " << pgc_model_ptr_->getMaxEccentricity() << " °" << std::endl;

    // Findout cellWidth and cellHeight that fit Pixcone layer
    int cellsWidth = pgc_model_ptr_->getIndexAt(cone_model.getEccentricityAt(cones_cpu.width / 2.0)) * 2;
    int cellsHeight = pgc_model_ptr_->getIndexAt(cone_model.getEccentricityAt(cones_cpu.height / 2.0)) * 2;
    cellsWidth -= cellsWidth % BLOCK_SIZE;
    cellsHeight -= cellsHeight % BLOCK_SIZE;

    if (cellsHeight <= 0 || cellsWidth <= 0) {
      std::cerr << "Parameter implies empty cone array" << std::endl;
      throw std::invalid_argument("Parameter implies empty cone array");
    }

    cells_cpu_.height = cellsHeight;
    cells_cpu_.width = cellsWidth;

    std::cout << "Pix parasol cells radius: " << cellsWidth / 2 << " cells" << std::endl;
    std::cout << "Pix parasol cells max eccentricity: " << pgc_model_ptr_->getEccentricityAt(cellsWidth / 2) << " °"
              << std::endl;
    std::cout << "parasol dimensions: " << cells_cpu_.width << " x " << cells_cpu_.height
              << std::endl;   //" pgc_model_ptr_->getTotalRadius()

    cells_cpu_.gcells.resize(cells_cpu_.width * cells_cpu_.height);

    Ganglionar cell;
    double r;
    double ganglionarExternalRadius;
    double ganglionarInternalRadius;

    // Default model
    for (int j = 0; j < cells_cpu_.height; j++) {
      for (int i = 0; i < cells_cpu_.width; i++) {
        r = polar_utils::getDistanceFromCenter(i, j, cells_cpu_.width, cells_cpu_.height);

        const auto parasol_angular_pose = pgc_model_ptr_->getEccentricityAt(r);
        const auto parasol_central_cone = cone_model.getIndexAt(parasol_angular_pose);
        const auto mgc_density = pgc_model_ptr_->getDensityAt(parasol_angular_pose);
        const auto cone_density = cone_model.getDensityAt(parasol_angular_pose);

        if (parasol_angular_pose < 0 || parasol_central_cone < 0) {
          cell.type = GC_RESPONSE_TYPE::NONE;
          cells_cpu_.gcells[i + j * cells_cpu_.width] = cell;
          continue;
        }

        ganglionarExternalRadius = std::sqrt(UNIT_DISK_SURFACE * (cone_density / (mgc_density)) * 8.0 / M_PI);   //
        ganglionarExternalRadius = MAX(0.5, ganglionarExternalRadius);
        ganglionarInternalRadius = MAX(0.5, 0.33 * ganglionarExternalRadius);

        cv::Vec2f direction = r == 0 ? cv::Vec2f(1, 0)
                                     : polar_utils::getDirectionFromCenter(
                                           cv::Point(i, j), cv::Size(cells_cpu_.width, cells_cpu_.height));
        cv::Point src_pos =
            polar_utils::getPosition(parasol_central_cone, cv::Size(cones_cpu.width, cones_cpu.height), direction);

        // check if cone is valid
        unsigned int cone_key = src_pos.x + src_pos.y * cones_cpu.width;
        if (cone_key >= cones_cpu.cones.size() || src_pos.x < 0 || src_pos.x >= cones_cpu.width || src_pos.y < 0 ||
            src_pos.y >= cones_cpu.height || cones_cpu.cones[cone_key].type == PHOTO_TYPE::NONE) {
          cell.type = GC_RESPONSE_TYPE::NONE;
          cells_cpu_.gcells[i + j * cells_cpu_.width] = cell;
          continue;
        }

        cell.center_x = src_pos.x;
        cell.center_y = src_pos.y;
        cell.extern_radius = ganglionarExternalRadius;
        cell.intern_radius = ganglionarInternalRadius;
        cell.type = i % 2 == 1 ? GC_RESPONSE_TYPE::ON : GC_RESPONSE_TYPE::OFF;
        cells_cpu_.gcells[i + j * cells_cpu_.width] = cell;

        if (j == cells_cpu_.height / 8) {
          description_map_["gc_external_radius_at_eighth"].push_back(ganglionarExternalRadius);
          description_map_["gc_external_radius_at_eighth_x"].push_back(i);
          description_map_["gc_external_radius_at_eighth_deg"].push_back(parasol_angular_pose);

          description_map_["gc_parasol_angular_eighth_pose"].push_back(parasol_angular_pose);
          description_map_["gc_parasol_angular_eighth_pose_x"].push_back(i);

          description_map_["gc_parasol_cone_eighth_pose"].push_back(parasol_central_cone);
          description_map_["gc_parasol_cone_eighth_pose_x"].push_back(i);
          description_map_["gc_parasol_cone_eighth_pose_deg"].push_back(parasol_angular_pose);
        } else if (j == cells_cpu_.height / 2) {
          description_map_["gcp_radius_at_eighth"].push_back(cone_density / mgc_density);
          // description_map_["gcp_radius_at_eighth"].push_back(next_parasol_central_cone - parasol_central_cone);
          description_map_["gcp_radius_at_eighth_x"].push_back(parasol_angular_pose);

          description_map_["gc_external_radius_at_half"].push_back(ganglionarExternalRadius);
          description_map_["gc_external_radius_at_half_x"].push_back(i);
          description_map_["gc_external_radius_at_half_deg"].push_back(parasol_angular_pose);

          description_map_["pgc_cones_in_receptive_at_half"].push_back(M_PI * ganglionarExternalRadius *
                                                                       ganglionarExternalRadius / UNIT_DISK_SURFACE);
          description_map_["pgc_cones_in_receptive_at_half_deg"].push_back(parasol_angular_pose);

          description_map_["gc_parasol_angular_half_pose"].push_back(parasol_angular_pose);
          description_map_["gc_parasol_angular_half_pose_x"].push_back(i);

          description_map_["gc_parasol_cone_half_pose"].push_back(parasol_central_cone);
          description_map_["gc_parasol_cone_half_pose_x"].push_back(i);
          description_map_["gc_parasol_cone_half_pose_deg"].push_back(parasol_angular_pose);

          description_map_["gcp_density"].push_back(pgc_model_ptr_->getDensityAt(parasol_angular_pose));
          description_map_["cone_density"].push_back(cone_model.getDensityAt(parasol_angular_pose));
          description_map_["gcp_cone_ratio"].push_back(pgc_model_ptr_->getDensityAt(parasol_angular_pose) /
                                                       cone_model.getDensityAt(parasol_angular_pose));
          description_map_["gcp_density_deg"].push_back(parasol_angular_pose);
        }
      }
    }
  }

  void plotGraphs();

  const GanglionarCells &pgcells() const { return cells_cpu_; }

 private:
  // Map containing info on the ganglionar cells properties
  std::map<std::string, std::vector<float>> description_map_;
  GanglionarCells cells_cpu_;
  std::unique_ptr<PGCellsModel> pgc_model_ptr_;
  Random random;
};