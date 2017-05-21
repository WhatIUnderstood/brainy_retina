#ifndef DIMENSIONS_H
#define DIMENSIONS_H

#include <ostream>

struct Size2D{
    Size2D(int x=0, int y=0):width(x),height(y){}
    bool operator == ( const Size2D& r){
        return height == r.height && width == r.width;
    }

    bool operator != ( const Size2D& r){
        return height != r.height || width != r.width;
    }

    int width;
    int height;
};

std::ostream &operator<<(std::ostream &os, Size2D const &m);

struct Point2D{
    Point2D(int x=0, int y=0):x(x),y(y){}
    int x;
    int y;

    bool operator ==(const Point2D  & other) const {return x==other.x && y==other.y;}
};

std::ostream &operator<<(std::ostream &os, Point2D const &m);

struct Point3D{
    Point3D(int x=0, int y=0, int z=0):x(x),y(y),z(z){}
    int x;
    int y;
    int z;

    bool operator ==(Point3D const & other){return x==other.x && y==other.y && z==other.z;}
};

std::ostream &operator<<(std::ostream &os, Point3D const &m);



#endif // DIMENSIONS_H

