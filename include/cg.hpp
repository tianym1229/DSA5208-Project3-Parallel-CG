#pragma once
#include <vector>
#include <string>
struct CSR { int n=0; std::vector<int> rowptr,col; std::vector<double> val; };
struct Result {std::vector<double> x; int iterations=0; double relative_residual=0, seconds=0; bool converged=false;};
CSR laplacian(int dim,int m);
std::vector<double> multiply(const CSR& a,const std::vector<double>& x);
Result conjugate_gradient(const CSR& a,const std::vector<double>& b,double tol,int max_iter);
double relative_error(const std::vector<double>& x,const std::vector<double>& truth);
double relative_residual(const CSR& a,const std::vector<double>& x,const std::vector<double>& b);
