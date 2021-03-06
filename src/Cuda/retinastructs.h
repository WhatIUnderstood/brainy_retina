#pragma once
#include <vector>
#include <cmath>

constexpr int BLOCK_SIZE = 32;

enum class PHOTO_TYPE
{
    S_CONE = 0,
    M_CONE = 1,
    L_CONE = 2,
    NONE = 3

};

enum class GC_RESPONSE_TYPE
{
    ON = 0,
    OFF = 1,
    NONE = 3
};

struct Ganglionar
{
    int center_x;
    int center_y;
    float intern_radius;
    float extern_radius;
    GC_RESPONSE_TYPE type; //0 ON, 1 OFF
};

struct Cone
{
    int center_x;
    int center_y;
    PHOTO_TYPE type; // 0 L cones, 1 M cones, 2
};

struct Cones
{
    std::vector<Cone> cones;
    int width = 0;
    int height = 0;
};

struct GanglionarCells
{
    std::vector<Ganglionar> gcells;
    int width = 0;
    int height = 0;
};

struct Point
{
    Point(int x, int y) : x(x), y(y) {}
    int x;
    int y;
};

constexpr double UNIT_DISK_SURFACE = M_PI * 0.5 * 0.5;
