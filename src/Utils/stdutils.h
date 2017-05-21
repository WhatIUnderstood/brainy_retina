#ifndef STDUTILS_H
#define STDUTILS_H

#include <vector>
#include <string>

class StdUtils
{
public:
    StdUtils();
    static std::vector<std::string> split(const std::string& val, char separator);
};

#endif // STDUTILS_H
