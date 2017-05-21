#include "dimensions.h"

std::ostream &operator<<(std::ostream &os, Size2D const &m) {
    return os << "( "<<m.width<<"x"<<m.height<<" )";
}
std::ostream &operator<<(std::ostream &os, Point2D const &m) {
    return os << "( "<<m.x<<","<<m.y<<" )";
}
std::ostream &operator<<(std::ostream &os, Point3D const &m) {
    return os << "( "<<m.x<<","<<m.y<<","<<m.z<<" )";
}

