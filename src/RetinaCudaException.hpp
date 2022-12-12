#pragma once
#include <exception>
#include <string>

class RetinaCudaException : public std::exception {
 public:
  RetinaCudaException(const std::string& msg) : msg_(msg) {}

  const char* what() const noexcept override { return msg_.c_str(); }

 private:
  std::string msg_;
};