#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <ostream>

class Logger
{
public:
    enum LOG_BEHAVIOUR{
        LOG_BEHAVIOUR_NEWLINE,
        LOG_BEHAVIOUR_APPEND,
        LOG_BEHAVIOUR_HIDDEN
    };

    Logger(std::ostream& stream=std::cout ) : m_file(stream) {behaviour = LOG_BEHAVIOUR_NEWLINE;}
    ~Logger(){
        switch (behaviour) {
        case LOG_BEHAVIOUR_APPEND:
            break;
        case LOG_BEHAVIOUR_NEWLINE:
            m_file<<std::endl;
            break;
        default:
            break;
        }
    }

    template <typename T>
    Logger &operator<<(const T &a) {
        switch (behaviour) {
        case LOG_BEHAVIOUR_APPEND:
        case LOG_BEHAVIOUR_NEWLINE:
            m_file<<a;
            break;
        case LOG_BEHAVIOUR_HIDDEN:
            break;
        default:
            break;
        }
        return *this;
    }

    Logger& operator()(LOG_BEHAVIOUR a = LOG_BEHAVIOUR_NEWLINE) {
        behaviour = a;
        return *this;
    }


protected:
    std::ostream& m_file;
private:
    LOG_BEHAVIOUR behaviour;
};


#define logtrace Logger()
#define logd Logger()
#define logw Logger()
#define logerr Logger(std::cerr)

#endif // LOGGER_H
