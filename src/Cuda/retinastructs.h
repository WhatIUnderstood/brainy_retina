#ifndef RETINA_STRUCTS_H
#define RETINA_STRUCTS_H

struct Cell{
    int center_x;
    int center_y;
    int intern_radius;
    int extern_radius;
    unsigned int type;//0 ON, 1 OFF
};



#endif // RETINA_STRUCTS_H
