#pragma once
#include <string>
#include "grid.hpp"

// Writes cell-centered ImageData (.vti) with nx*ny tuples on each rank.
// Open all *_rank*.vti files in ParaView and use "Append Datasets" to view the whole field.
void write_vti_cellcentered(const std::string& path, const Cart2D& cart, const Grid& g);

// (optional) CSV helper
void write_csv(const std::string& path, const std::string& header, const std::string& line);
