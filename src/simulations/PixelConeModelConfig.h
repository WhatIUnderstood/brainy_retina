#pragma once

/**
 * @brief Camera intrinsic parameters.
 */
struct PixelConeModelConfig {
  double camera_hfov = 0;   // degree
  int camera_width = 0;
  int camera_height = 0;
};
