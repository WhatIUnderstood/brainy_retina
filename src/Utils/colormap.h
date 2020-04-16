#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <exception>

namespace colormap
{

using ColorMap = std::vector<cv::Mat>;

enum class COLORMAP_TYPE
{
    BLUE_GREEN_RED // blue -> green -> red map
};

ColorMap _buildColorMap_BLUE_GREEN_RED();

inline cv ::Mat convertToLUT(const ColorMap &map)
{
    cv::Mat channels[] = {map[0], map[1], map[2]};
    cv ::Mat lut; // Créer une table de recherche
    cv::merge(channels, 3, lut);
    return lut;
}

/**
 * @brief Build a colormap with the given type
 *
 * @param type
 * @return ColorMap mapping for each channel (blue, green, red)
 */
inline ColorMap buildColorMap(COLORMAP_TYPE type)
{
    switch (type)
    {
    case COLORMAP_TYPE::BLUE_GREEN_RED:
        return colormap::_buildColorMap_BLUE_GREEN_RED();
        break;
    default:
        throw std::invalid_argument("Unknown colormap type");
        break;
    }
}

inline cv::Mat buildColorMapImage(cv::Mat lut, int width, int height)
{
    cv::Mat color_map(height, width, CV_8UC3);
    for (unsigned int i = 0; i < width; i++)
    {
        for (unsigned int j = 0; j < height; j++)
        {
            color_map.at<cv::Vec3b>(j, i) = cv::Vec3b(i / (float)width * 255, i / (float)width * 255, i / (float)width * 255);
        }
    }

    cv::Mat color_map_img;
    cv::LUT(color_map, lut, color_map_img);
    return color_map_img;
}

/**
 * @brief Reverse the given color mapping
 *
 * @param maps
 * @return ColorMap
 */
inline ColorMap reverseColorMap(const ColorMap &maps)
{
    ColorMap output;
    for (const auto &map : maps)
    {
        cv::Mat dst = map; // dst must be a different Mat
        cv::flip(map, dst, 0);
        output.push_back(dst);
    }
    return output;
}

inline ColorMap mirrorColorMap(const ColorMap &maps)
{
    ColorMap output;
    for (const auto &map : maps)
    {
        cv::Mat dst(128, 1, CV_8UC1);
        cv::resize(map, dst, dst.size(), cv::INTER_CUBIC);

        cv::Mat dst_flip;
        cv::flip(dst, dst_flip, 0);

        cv::Mat final_dst; // dst must be a different Mat
        cv::vconcat(dst, dst_flip, final_dst);
        output.push_back(final_dst);
    }
    return output;
}

// void applyColorMap(const cv::Mat &input_mat, cv::Mat &output_mat)
// {
//     cv::LUT(frameRetina, color_mapping[0], r1);
//     cv::LUT(frameRetina, color_mapping[1], r2);
//     cv::LUT(frameRetina, color_mapping[2], r3);
//     ColorMap planes;
//     planes.push_back(r1);
//     planes.push_back(r2);
//     planes.push_back(r3);
//     cv::merge(planes, cv_cm_img0);
// }

inline ColorMap _buildColorMap_BLUE_GREEN_RED()
{
    //Create custom color map
    cv::Mat b(256, 1, CV_8UC1);
    cv::Mat g(256, 1, CV_8UC1);
    cv::Mat r(256, 1, CV_8UC1);
    cv::Mat r1, r2, r3;

    float color_shift = 64;

    for (int i = 0; i < 256; i++)
    {
        if (i < color_shift)
        {
            b.at<uchar>(i) = 220;
            g.at<uchar>(i) = 20 + i / color_shift * 200;
            r.at<uchar>(i) = 20;
        }
        else if (i < 126)
        {
            b.at<uchar>(i) = 220 - (i - color_shift) / color_shift * 200;
            g.at<uchar>(i) = 220;
            r.at<uchar>(i) = 0;
        }
        else if (i < 190)
        {
            b.at<uchar>(i) = 20;
            g.at<uchar>(i) = 220;
            r.at<uchar>(i) = 20 + (i - 126) / color_shift * 200;
        }
        else
        {
            b.at<uchar>(i) = 20;
            g.at<uchar>(i) = 220 - (i - 190) / color_shift * 200;
            r.at<uchar>(i) = 220;
        }
    }
    ColorMap channels = {b, g, r};
    return channels;
}

// std::vector<cv::Mat> buildColorMapSimetrical()
// {
//     //Create custom color map
//     cv::Mat b(256, 1, CV_8UC1);
//     cv::Mat g(256, 1, CV_8UC1);
//     cv::Mat r(256, 1, CV_8UC1);
//     cv::Mat r1, r2, r3;

//     float color_shift = 32;

//     for (int i = 0; i < 256; i++)
//     {
//         if (i < color_shift)
//         {
//             b.at<uchar>(i) = 220;
//             g.at<uchar>(i) = 20 + i / color_shift * 200;
//             r.at<uchar>(i) = 20;
//         }
//         else if (i < 2 * color_shift)
//         {
//             b.at<uchar>(i) = 220 - (i - color_shift) / color_shift * 200;
//             g.at<uchar>(i) = 220;
//             r.at<uchar>(i) = 0;
//         }
//         else if (i < 3 * color_shift)
//         {
//             b.at<uchar>(i) = 20;
//             g.at<uchar>(i) = 220;
//             r.at<uchar>(i) = 20 + (i - 2 * color_shift) / color_shift * 200;
//         }
//         else if (i < 4 * color_shift)
//         {
//             b.at<uchar>(i) = 20;
//             g.at<uchar>(i) = 220 - (i - 3 * color_shift) / color_shift * 200;
//             r.at<uchar>(i) = 220;
//         }
//         else
//         {
//             //symetrical
//             b.at<uchar>(i) = b.at<uchar>(255 - i);
//             g.at<uchar>(i) = g.at<uchar>(255 - i);
//             r.at<uchar>(i) = r.at<uchar>(255 - i);
//         }
//     }
//     std::vector<cv::Mat> channels = {b, g, r};
//     return channels;
// };
} // namespace colormap