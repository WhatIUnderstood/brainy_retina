#pragma once
#include <string>
#include <algorithm>
#include <cctype>

namespace utils
{
inline bool isNumber(const std::string &s)
{
    return !s.empty() && std::find_if(s.begin(),
                                      s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}
} // namespace utils