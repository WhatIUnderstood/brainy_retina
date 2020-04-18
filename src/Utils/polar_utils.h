#pragma once

#include <opencv2/opencv.hpp>

namespace polar_utils
{
inline cv::Vec2f getDirectionFromCenter(cv::Point topLeftPosition, cv::Size size)
{
    return cv::normalize(cv::Vec2f(topLeftPosition.x - size.width / 2.0, topLeftPosition.y - size.height / 2.0));
}

inline cv::Point getPosition(double a_distance_from_center, cv::Size a_size, cv::Vec2f direction)
{
    cv::Vec2f normalizedDirection = cv::normalize(direction);
    double x_center = a_distance_from_center * normalizedDirection[0];
    double y_center = a_distance_from_center * normalizedDirection[1];

    return cv::Point(round(x_center + a_size.width / 2.0), round(y_center + a_size.height / 2.0));
}

inline double getDistanceFromCenter(double pix_x, double pix_y, double width, double height)
{
    return sqrt(std::pow(pix_x - width / 2.0, 2) + std::pow(pix_y - height / 2.0, 2));
}
} // namespace polar_utils