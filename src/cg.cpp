#include "cg.hpp"
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <limits>
#include <algorithm>

CSR laplacian(int dim,int m) {
 if(dim!=2 && dim!=3) throw std::invalid_argument("dim must be 2 or 3");
 if(m<2) throw std::invalid_argument("grid edge m must be >= 2");
 long long n64=1; for(int k=0;k<dim;++k){if(n64>(std::numeric_limits<int>::max()-1LL)/m)throw std::invalid_argument("matrix dimension overflow"); n64*=m;}
 if(n64>std::numeric_limits<int>::max()-1) throw std::invalid_argument("grid exceeds 32-bit CSR indexing");
 CSR a; a.n=(int)n64; a.rowptr.reserve(a.n+1); a.rowptr.push_back(0);
 for(int i=0;i<a.n;++i){
   int x=i%m,y=(i/m)%m,z=(dim==3)?i/(m*m):0;
   std::vector<std::pair<int,double>> entries;
   entries.push_back({i,dim==2?4.0:6.0});
   if(x>0) entries.push_back({i-1,-1});
   if(x+1<m) entries.push_back({i+1,-1});
   if(y>0) entries.push_back({i-m,-1});
   if(y+1<m) entries.push_back({i+m,-1});
   if(dim==3){ if(z>0) entries.push_back({i-m*m,-1}); if(z+1<m) entries.push_back({i+m*m,-1}); }
   std::sort(entries.begin(),entries.end());
   for(auto e:entries){a.col.push_back(e.first);a.val.push_back(e.second);}
   a.rowptr.push_back((int)a.col.size());
 }
 return a;
}
std::vector<double> multiply(const CSR& a,const std::vector<double>& x){
 if((int)x.size()!=a.n) throw std::invalid_argument("SpMV vector size mismatch");
 std::vector<double> y(a.n,0);
 for(int i=0;i<a.n;++i) for(int j=a.rowptr[i];j<a.rowptr[i+1];++j) y[i]+=a.val[j]*x[a.col[j]];
 return y;
}
static double dot(const std::vector<double>& a,const std::vector<double>& b){double s=0;for(size_t i=0;i<a.size();++i)s+=a[i]*b[i];return s;}
static double norm(const std::vector<double>& a){return std::sqrt(dot(a,a));}
Result conjugate_gradient(const CSR& a,const std::vector<double>& b,double tol,int max_iter){
 if((int)b.size()!=a.n || tol<=0 || max_iter<1) throw std::invalid_argument("invalid CG arguments");
 Result out;out.x.assign(a.n,0);std::vector<double> r=b,p=r;
 auto t0=std::chrono::steady_clock::now();
 double bnorm=norm(b),rr=dot(r,r);
 if(bnorm==0){out.converged=true;out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();return out;}
 for(int k=0;k<max_iter;++k){
   auto ap=multiply(a,p);double pap=dot(p,ap);
   if(!(pap>0) || !std::isfinite(pap)) throw std::runtime_error("CG breakdown: nonpositive p^T A p");
   double alpha=rr/pap;
   for(int i=0;i<a.n;++i){out.x[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
   double rrnew=dot(r,r);out.iterations=k+1;
   if(!std::isfinite(rrnew)) throw std::runtime_error("CG breakdown: nonfinite residual");
   if(std::sqrt(rrnew)/bnorm<tol){out.converged=true;break;}
   if(rr==0) throw std::runtime_error("CG breakdown: zero previous residual");
   double beta=rrnew/rr;
   for(int i=0;i<a.n;++i)p[i]=r[i]+beta*p[i];
   rr=rrnew;
 }
 out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
 out.relative_residual=relative_residual(a,out.x,b);
 if(out.relative_residual>=tol) out.converged=false;
 return out;
}
double relative_error(const std::vector<double>& x,const std::vector<double>& truth){
 if(x.size()!=truth.size())throw std::invalid_argument("relative_error size mismatch");
 double numerator=0,denominator=0;for(size_t i=0;i<x.size();++i){double d=x[i]-truth[i];numerator+=d*d;denominator+=truth[i]*truth[i];}
 return std::sqrt(numerator)/std::max(std::sqrt(denominator),std::numeric_limits<double>::min());
}
double relative_residual(const CSR& a,const std::vector<double>& x,const std::vector<double>& b){
 auto ax=multiply(a,x);double num=0,den=0;
 for(int i=0;i<a.n;++i){double d=ax[i]-b[i];num+=d*d;den+=b[i]*b[i];}
 return std::sqrt(num)/std::max(std::sqrt(den),std::numeric_limits<double>::min());
}
