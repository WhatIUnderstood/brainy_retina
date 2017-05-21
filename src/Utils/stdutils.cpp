#include "stdutils.h"

StdUtils::StdUtils()
{

}

std::vector<std::__cxx11::string> StdUtils::split(const std::__cxx11::string &val, char separator)
{
    std::vector<std::__cxx11::string> out;
    auto pos = val.find(separator);
    int prev = 0;
    while(pos != std::string::npos){
        out.push_back(val.substr(prev,pos-prev+1));
        prev = pos;
        pos = val.find(separator);
    }
    return out;
}
