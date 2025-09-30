#include "options.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

Options parse_options(int argc, char** argv){
  Options o;
  for (int i=1; i<argc; ++i){
    std::string k = argv[i];
    auto next = [&](int& i){ if(i+1<argc) return std::string(argv[++i]); std::cerr<<"Missing value for "<<k<<"\n"; std::exit(1); };
    if(k=="--N") o.N = std::stoi(next(i));
    else if(k=="--lambda") o.lambda = std::stod(next(i));
    else if(k=="--rtol") o.rtol = std::stod(next(i));
    else if(k=="--gmres_restart") o.gmres_restart = std::stoi(next(i));
    else if(k=="--max_newton") o.max_newton = std::stoi(next(i));
    else if(k=="--max_gmres_it") o.max_gmres_it = std::stoi(next(i));
    else if(k=="--omp_threads") o.omp_threads = std::stoi(next(i));
    else if(k=="--output_prefix") o.output_prefix = next(i);
    else if(k=="--help"){
      std::cout<<"Usage: [--N 512] [--lambda 3.0] [--rtol 1e-8] [--gmres_restart 50] [--max_newton 30] [--max_gmres_it 200] [--omp_threads 0] [--output_prefix data/run]\n";
      std::exit(0);
    }
  }
  return o;
}
