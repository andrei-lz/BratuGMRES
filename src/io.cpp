#include "io.hpp"
#include <fstream>
#include <iomanip>
#include <string>

// ASCII ImageData (.vti), cell-centered scalar 'u' with nx*ny tuples.
// Origin is placed at the first cell center of this subdomain.
void write_vti_cellcentered(const std::string& path, const Cart2D& /*cart*/, const Grid& g){
  const int nx = g.nx, ny = g.ny;
  const int nxh = nx + 2;

  // For cell-centered ImageData, we still set WholeExtent in *point* indices.
  // Using 0..nx and 0..ny is fine; CellData will have nx*ny tuples.
  const int i0=0, i1=nx;
  const int j0=0, j1=ny;
  const int k0=0, k1=0;

  // Place origin at first cell center of the local block.
  // Global offset (gx0,gy0) counts interior cells before this block.
  const double ox = (g.gx0 + 0.5) * g.h;
  const double oy = (g.gy0 + 0.5) * g.h;
  const double oz = 0.0;

  std::ofstream f(path);
  f << std::setprecision(17);
  f << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
  f << "  <ImageData WholeExtent=\"" << i0 << " " << i1 << " "
                                   << j0 << " " << j1 << " "
                                   << k0 << " " << k1 << "\" "
    << "Origin=\"" << ox << " " << oy << " " << oz << "\" "
    << "Spacing=\"" << g.h << " " << g.h << " 1\">\n";
  f << "    <Piece Extent=\"" << i0 << " " << i1 << " "
                              << j0 << " " << j1 << " "
                              << k0 << " " << k1 << "\">\n";

  // We store u on cells (nx*ny tuples).
  f << "      <CellData Scalars=\"u\">\n";
  f << "        <DataArray type=\"Float64\" Name=\"u\" format=\"ascii\">\n";
  for(int j=1; j<=ny; ++j){
    for(int i=1; i<=nx; ++i){
      f << g.u[j*nxh + i] << " ";
    }
    f << "\n";
  }
  f << "        </DataArray>\n";
  f << "      </CellData>\n";

  // (No PointData arrays.)
  f << "    </Piece>\n";
  f << "  </ImageData>\n";
  f << "</VTKFile>\n";
}

void write_csv(const std::string& path, const std::string& header, const std::string& line){
  std::ofstream f(path, std::ios::app);
  if(f.tellp()==0 && !header.empty()) f<<header<<"\n";
  if(!line.empty()) f<<line<<"\n";
}
