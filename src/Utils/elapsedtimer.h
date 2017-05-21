#ifndef ELAPSEDTIMER_H
#define ELAPSEDTIMER_H

#include <iostream>
#include <chrono>
#include <ctime>

typedef std::chrono::system_clock Time;

class ElapsedTimer
{
public:
    ElapsedTimer();
    void start();
    double elapsed();
    double restart();
private:
    std::chrono::time_point<std::chrono::system_clock> start_t, end_t;
};

#endif // ELAPSEDTIMER_H
