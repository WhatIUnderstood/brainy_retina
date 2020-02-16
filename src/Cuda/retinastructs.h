#ifndef RETINA_STRUCTS_H
#define RETINA_STRUCTS_H

enum PHOTO_TYPE{
    PHOTO_R = 0,
    PHOTO_G = 1,
    PHOTO_B = 2
};

struct Ganglionar{
    int center_x;
    int center_y;
    int intern_radius;
    int extern_radius;
    unsigned int type;//0 ON, 1 OFF
};

struct Cone{
    int center_x;
    int center_y;
    int type;

};

struct Point{
    Point(int x, int y):x(x),y(y){}
    int x;
    int y;
};



#endif // RETINA_STRUCTS_H
