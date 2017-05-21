#include "elapsedtimer.h"

ElapsedTimer::ElapsedTimer()
{

}

void ElapsedTimer::start()
{
    start_t = std::chrono::system_clock::now();
}

double ElapsedTimer::elapsed()
{
    end_t = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_t-start_t;
    return elapsed_seconds.count();
}

double ElapsedTimer::restart()
{
    end_t = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_t-start_t;
    return elapsed_seconds.count();
    start_t = end_t;
}
