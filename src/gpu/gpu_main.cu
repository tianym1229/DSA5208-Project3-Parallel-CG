#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t s=(x); if(s!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(s)); } while(0)
#define BLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+" failed (cuBLAS status "+std::to_string((int)s)+")"); } while(0)
#define SPARSE_CHECK(x) do { cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+" failed (cuSPARSE status "+std::to_string((int)s)+")"); } while(0)

struct CSR { int n=0; std::vector<int> row,col; std::vector<double> val; };
static double truth_at(int i,int dim,int m,const std::string& kind) {
    if(kind=="ones") return 1.0;
    if(kind=="benchmark") {
        unsigned long long h=static_cast<unsigned long long>(i)+0x9e3779b97f4a7c15ULL;
        h=(h^(h>>30))*0xbf58476d1ce4e5b9ULL;
        h=(h^(h>>27))*0x94d049bb133111ebULL;
        h^=h>>31;
        return 0.5+static_cast<double>(h>>11)*(1.0/9007199254740992.0);
    }
    int x=i%m,y=(i/m)%m,z=dim==3?i/(m*m):0;
    double pi=std::acos(-1.0);
    double t=std::sin(pi*(x+1)/(m+1.0))*std::sin(pi*(y+1)/(m+1.0));
    if(dim==3)t*=std::sin(pi*(z+1)/(m+1.0));
    return t;
}
static CSR generate(int dim,int m) {
    if((dim!=2&&dim!=3)||m<2) throw std::runtime_error("invalid dimension/grid");
    long long n=1;
    for(int d=0;d<dim;++d){if(n>(std::numeric_limits<int>::max()-1LL)/m)throw std::runtime_error("grid overflow");n*=m;}
    CSR a; a.n=(int)n; a.row.reserve(a.n+1);a.row.push_back(0);
    for(int i=0;i<a.n;++i){
        int x=i%m,y=(i/m)%m,z=dim==3?i/(m*m):0;
        std::vector<std::pair<int,double>> e{{i,dim==2?4.0:6.0}};
        if(x>0)e.push_back({i-1,-1});if(x+1<m)e.push_back({i+1,-1});
        if(y>0)e.push_back({i-m,-1});if(y+1<m)e.push_back({i+m,-1});
        if(dim==3){if(z>0)e.push_back({i-m*m,-1});if(z+1<m)e.push_back({i+m*m,-1});}
        std::sort(e.begin(),e.end());
        for(const auto& v:e){a.col.push_back(v.first);a.val.push_back(v.second);}
        if(a.col.size()>(size_t)std::numeric_limits<int>::max())throw std::runtime_error("CSR nnz overflow");
        a.row.push_back((int)a.col.size());
    }
    return a;
}
static std::vector<double> multiply(const CSR& a,const std::vector<double>& x){
    std::vector<double> y(a.n,0);
    for(int i=0;i<a.n;++i)for(int j=a.row[i];j<a.row[i+1];++j)y[i]+=a.val[j]*x[a.col[j]];
    return y;
}
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n=0){if(n)CUDA_CHECK(cudaMalloc((void**)&p,n*sizeof(T)));}
    ~Device(){if(p)cudaFree(p);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    T* get()const{return p;}
};
struct Event {
    cudaEvent_t e=nullptr;
    Event(){CUDA_CHECK(cudaEventCreate(&e));}
    ~Event(){if(e)cudaEventDestroy(e);}
    Event(const Event&)=delete;Event& operator=(const Event&)=delete;
};
struct Blas {
    cublasHandle_t h=nullptr;
    Blas(){BLAS_CHECK(cublasCreate(&h));BLAS_CHECK(cublasSetPointerMode(h,CUBLAS_POINTER_MODE_HOST));}
    ~Blas(){if(h)cublasDestroy(h);}
};
struct Sparse {
    cusparseHandle_t h=nullptr;
    Sparse(){SPARSE_CHECK(cusparseCreate(&h));}
    ~Sparse(){if(h)cusparseDestroy(h);}
};
struct Matrix {
    cusparseSpMatDescr_t d=nullptr;
    Matrix(int n,int nnz,int* row,int* col,double* val){
        SPARSE_CHECK(cusparseCreateCsr(&d,n,n,nnz,row,col,val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    }
    ~Matrix(){if(d)cusparseDestroySpMat(d);}
};
struct Vector {
    cusparseDnVecDescr_t d=nullptr;
    Vector(int n,double* p){SPARSE_CHECK(cusparseCreateDnVec(&d,n,p,CUDA_R_64F));}
    ~Vector(){if(d)cusparseDestroyDnVec(d);}
};
static double seconds(std::chrono::steady_clock::time_point t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
int main(int argc,char** argv){
try {
    int dim=2,m=16,maxiter=10000;double tol=1e-8;std::string solution="benchmark";
    for(int i=1;i<argc;++i){
        std::string k=argv[i];if(i+1>=argc)throw std::runtime_error("missing value for "+k);
        std::string v=argv[++i];
        if(k=="--dim")dim=std::stoi(v);else if(k=="--grid")m=std::stoi(v);
        else if(k=="--tol")tol=std::stod(v);else if(k=="--max-iters")maxiter=std::stoi(v);
        else if(k=="--solution")solution=v;else throw std::runtime_error("unknown option "+k);
    }
    if((dim!=2&&dim!=3)||m<2||!std::isfinite(tol)||tol<=0||maxiter<1||(solution!="benchmark"&&solution!="smooth"&&solution!="ones"))throw std::runtime_error("invalid arguments");
    auto whole=std::chrono::steady_clock::now();
    CSR a=generate(dim,m);int n=a.n,nnz=(int)a.val.size();
    std::vector<double> truth(n);for(int i=0;i<n;++i)truth[i]=truth_at(i,dim,m,solution);
    std::vector<double> b(n,0);
    // Match the MPI RHS accumulation order, including the sorted CSR order.
    for(int i=0;i<n;++i)for(int j=a.row[i];j<a.row[i+1];++j)b[i]+=a.val[j]*truth_at(a.col[j],dim,m,solution);
    Device<int> dr(n+1),dc(nnz);Device<double> dv(nnz),db(n),dx(n),dp(n),dres(n),dap(n);
    auto transfer_start=std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(dr.get(),a.row.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc.get(),a.col.data(),nnz*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.get(),a.val.data(),nnz*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.get(),b.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp.get(),b.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dres.get(),b.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dx.get(),0,n*sizeof(double)));
    CUDA_CHECK(cudaDeviceSynchronize());
    double h2d_s=seconds(transfer_start);
    Blas blas;Sparse sparse;Matrix mat(n,nnz,dr.get(),dc.get(),dv.get());
    Vector pvec(n,dp.get()),apvec(n,dap.get());
    const double one=1.0,zero=0.0;
    size_t buffer_size=0;
    SPARSE_CHECK(cusparseSpMV_bufferSize(sparse.h,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,mat.d,pvec.d,&zero,apvec.d,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,&buffer_size));
    Device<unsigned char> buffer(buffer_size);
    auto spmv=[&](){SPARSE_CHECK(cusparseSpMV(sparse.h,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,mat.d,pvec.d,&zero,apvec.d,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,buffer.get()));};
    Event ev0,ev1;CUDA_CHECK(cudaEventRecord(ev0.e));
    double bb=0;BLAS_CHECK(cublasDdot(blas.h,n,db.get(),1,db.get(),1,&bb));
    double rr=bb;int iterations=0;bool converged=(bb==0);
    for(int k=0;k<maxiter&&!converged;++k){
        spmv();
        double pap=0;BLAS_CHECK(cublasDdot(blas.h,n,dp.get(),1,dap.get(),1,&pap));
        if(!(pap>0)||!std::isfinite(pap))throw std::runtime_error("CG breakdown: nonpositive pAp");
        double alpha=rr/pap,minus_alpha=-alpha;
        BLAS_CHECK(cublasDaxpy(blas.h,n,&alpha,dp.get(),1,dx.get(),1));
        BLAS_CHECK(cublasDaxpy(blas.h,n,&minus_alpha,dap.get(),1,dres.get(),1));
        double rrnew=0;BLAS_CHECK(cublasDdot(blas.h,n,dres.get(),1,dres.get(),1,&rrnew));
        iterations=k+1;
        if(!std::isfinite(rrnew))throw std::runtime_error("CG breakdown: nonfinite residual");
        if(std::sqrt(rrnew/bb)<tol){converged=true;break;}
        if(!(rr>0))throw std::runtime_error("CG breakdown: zero previous residual");
        double beta=rrnew/rr;
        BLAS_CHECK(cublasDscal(blas.h,n,&beta,dp.get(),1));
        BLAS_CHECK(cublasDaxpy(blas.h,n,&one,dres.get(),1,dp.get(),1));
        rr=rrnew;
    }
    CUDA_CHECK(cudaEventRecord(ev1.e));CUDA_CHECK(cudaEventSynchronize(ev1.e));
    float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,ev0.e,ev1.e));double solver_s=ms/1000.0;
    auto d2h_start=std::chrono::steady_clock::now();
    std::vector<double> x(n);CUDA_CHECK(cudaMemcpy(x.data(),dx.get(),n*sizeof(double),cudaMemcpyDeviceToHost));
    double d2h_s=seconds(d2h_start);
    auto val_start=std::chrono::steady_clock::now();
    auto ax=multiply(a,x);
    long double err=0,res=0,tnorm=0,bnorm=0;
    for(int i=0;i<n;++i){long double d=(long double)x[i]-truth[i],q=(long double)ax[i]-b[i];err+=d*d;res+=q*q;tnorm+=(long double)truth[i]*truth[i];bnorm+=(long double)b[i]*b[i];}
    double relerr=(double)(std::sqrt(err)/std::max(std::sqrt(tnorm),(long double)std::numeric_limits<double>::min()));
    double relres=(double)(std::sqrt(res)/std::max(std::sqrt(bnorm),(long double)std::numeric_limits<double>::min()));
    double validation_s=seconds(val_start);
    if(!(relres<tol))converged=false;
    double total_s=seconds(whole);
    std::cout<<std::setprecision(12);
    std::cout<<"dim,grid,n,nnz,solution,iterations,solver_runtime_s,h2d_s,d2h_s,validation_s,end_to_end_s,relative_residual,relative_error,converged\n";
    std::cout<<dim<<','<<m<<','<<n<<','<<nnz<<','<<solution<<','<<iterations<<','<<solver_s<<','<<h2d_s<<','<<d2h_s<<','<<validation_s<<','<<total_s<<','<<relres<<','<<relerr<<','<<(converged?1:0)<<'\n';
    return converged?0:2;
} catch(const std::exception& e){std::cerr<<"GPU CG ERROR: "<<e.what()<<'\n';return 1;}
}
