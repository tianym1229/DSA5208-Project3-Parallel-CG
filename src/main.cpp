#include "cg.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <string>
#include <limits>
int main(int argc,char** argv){
 try{
 int dim=2,m=16,maxiter=10000;double tol=1e-8;std::string solution="smooth";
 for(int i=1;i<argc;++i){std::string a=argv[i];if(i+1>=argc)throw std::invalid_argument("missing value for "+a);std::string v=argv[++i];
 if(a=="--dim")dim=std::stoi(v);else if(a=="--grid")m=std::stoi(v);else if(a=="--tol")tol=std::stod(v);else if(a=="--max-iters")maxiter=std::stoi(v);else if(a=="--solution")solution=v;else throw std::invalid_argument("unknown option "+a);
 }
 if(solution!="ones" && solution!="smooth" && solution!="benchmark")throw std::invalid_argument("solution must be ones, smooth or benchmark");
 auto A=laplacian(dim,m);std::vector<double> truth(A.n);
 for(int i=0;i<A.n;++i){
 if(solution=="ones")truth[i]=1.0;
 else if(solution=="benchmark") { // Deterministic hash, independent of MPI partitioning.
   unsigned long long h=static_cast<unsigned long long>(i)+0x9e3779b97f4a7c15ULL;
   h=(h^(h>>30))*0xbf58476d1ce4e5b9ULL;
   h=(h^(h>>27))*0x94d049bb133111ebULL;
   h^=h>>31;
   truth[i]=0.5+static_cast<double>(h>>11)*(1.0/9007199254740992.0);
 }
 else {int x=i%m,y=(i/m)%m,z=(dim==3)?i/(m*m):0;double pi=std::acos(-1.0);
 truth[i]=std::sin(pi*(x+1)/(m+1.0))*std::sin(pi*(y+1)/(m+1.0));
 if(dim==3)truth[i]*=std::sin(pi*(z+1)/(m+1.0));
 }
 }
 auto b=multiply(A,truth);auto res=conjugate_gradient(A,b,tol,maxiter);
 std::cout<<std::setprecision(12)<<"dim,grid,n,nnz,solution,iterations,runtime_s,relative_residual,relative_error,converged\n";
 std::cout<<dim<<','<<m<<','<<A.n<<','<<A.val.size()<<','<<solution<<','<<res.iterations<<','<<res.seconds<<','<<res.relative_residual<<','<<relative_error(res.x,truth)<<','<<(res.converged?1:0)<<'\n';
 return res.converged?0:2;
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
