#pragma once
#include <chrono>
struct Timer {
  std::chrono::high_resolution_clock::time_point t0;
  void tic(){ t0 = std::chrono::high_resolution_clock::now(); }
  double toc() const {
    using namespace std::chrono;
    return duration<double>(high_resolution_clock::now()-t0).count();
  }
};
