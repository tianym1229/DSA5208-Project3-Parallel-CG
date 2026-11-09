#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct LocalCSR {
    int begin=0, end=0, n=0;
    std::vector<int> rowptr{0}, col;
    std::vector<double> val;
};

static int owner(int global, int n, int p) {
    // Block partition, first n%p ranks own one extra row.
    int q=n/p, r=n%p;
    if (global < (q+1)*r) return global/(q+1);
    return r+(global-(q+1)*r)/q;
}
static int start(int rank,int n,int p){return rank*(n/p)+std::min(rank,n%p);}
static double truth_at(int i,int dim,int m,const std::string& kind){
    if(kind=="ones") return 1.0;
    if(kind=="benchmark") {
        unsigned long long h=static_cast<unsigned long long>(i)+0x9e3779b97f4a7c15ULL;
        h=(h^(h>>30))*0xbf58476d1ce4e5b9ULL;
        h=(h^(h>>27))*0x94d049bb133111ebULL;
        h^=h>>31;
        return 0.5+static_cast<double>(h>>11)*(1.0/9007199254740992.0);
    }
    const int x=i%m,y=(i/m)%m,z=(dim==3)?i/(m*m):0;
    const double pi=std::acos(-1.0);
    double t=std::sin(pi*(x+1)/(m+1.0))*std::sin(pi*(y+1)/(m+1.0));
    if(dim==3) t*=std::sin(pi*(z+1)/(m+1.0));
    return t;
}
static LocalCSR generate(int dim,int m,int rank,int size){
    long long n64=1;for(int d=0;d<dim;++d){if(n64>(std::numeric_limits<int>::max()-1LL)/m)throw std::runtime_error("matrix dimension overflow");n64*=m;}
    if(n64>std::numeric_limits<int>::max()-1)throw std::runtime_error("matrix too large for 32-bit indices");
    LocalCSR a;a.n=(int)n64;a.begin=start(rank,a.n,size);a.end=start(rank+1,a.n,size);
    for(int i=a.begin;i<a.end;++i){
        const int x=i%m,y=(i/m)%m,z=(dim==3)?i/(m*m):0;
        std::vector<std::pair<int,double>> e{{i,dim==2?4.0:6.0}};
        if(x>0) e.push_back({i-1,-1});
        if(x+1<m) e.push_back({i+1,-1});
        if(y>0) e.push_back({i-m,-1});
        if(y+1<m) e.push_back({i+m,-1});
        if(dim==3){if(z>0)e.push_back({i-m*m,-1});if(z+1<m)e.push_back({i+m*m,-1});}
        std::sort(e.begin(),e.end());
        for(const auto& v:e){a.col.push_back(v.first);a.val.push_back(v.second);}
        a.rowptr.push_back((int)a.col.size());
    }
    return a;
}

struct Halo {
    std::vector<int> recvcounts,sendcounts,recvdisp,senddisp,remote_ids,send_local;
    std::vector<double> incoming,outgoing;
    std::unordered_map<int,int> ghost_index;
    int rank=0,size=1;
    Halo(const LocalCSR& a,int rank_,int size_):rank(rank_),size(size_){
        std::vector<std::vector<int>> requested(size);
        for(int c:a.col)if(c<a.begin || c>=a.end)requested[owner(c,a.n,size)].push_back(c);
        recvcounts.resize(size);sendcounts.resize(size);
        recvdisp.resize(size);senddisp.resize(size);
        for(int peer=0;peer<size;++peer){
            auto& v=requested[peer];std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());
            recvcounts[peer]=(int)v.size();
        }
        MPI_Alltoall(recvcounts.data(),1,MPI_INT,sendcounts.data(),1,MPI_INT,MPI_COMM_WORLD);
        int nr=0,ns=0;
        for(int peer=0;peer<size;++peer){recvdisp[peer]=nr;nr+=recvcounts[peer];senddisp[peer]=ns;ns+=sendcounts[peer];}
        remote_ids.resize(nr);send_local.resize(ns);
        for(int peer=0;peer<size;++peer)
            std::copy(requested[peer].begin(),requested[peer].end(),remote_ids.begin()+recvdisp[peer]);
        std::vector<int> requests_from_others(ns);
        MPI_Alltoallv(remote_ids.data(),recvcounts.data(),recvdisp.data(),MPI_INT,
                      requests_from_others.data(),sendcounts.data(),senddisp.data(),MPI_INT,MPI_COMM_WORLD);
        for(int i=0;i<ns;++i){
            if(requests_from_others[i]<a.begin || requests_from_others[i]>=a.end)
                throw std::runtime_error("invalid halo request ownership");
            send_local[i]=requests_from_others[i]-a.begin;
        }
        for(int i=0;i<nr;++i)ghost_index[remote_ids[i]]=i;
        incoming.resize(nr);outgoing.resize(ns);
    }
    void pack(const std::vector<double>& x){
        for(size_t i=0;i<send_local.size();++i)outgoing[i]=x[send_local[i]];
    }
    void post(std::vector<MPI_Request>& requests){
        requests.clear();requests.reserve(2*size);
        for(int peer=0;peer<size;++peer)if(recvcounts[peer]){
            MPI_Request r;MPI_Irecv(incoming.data()+recvdisp[peer],recvcounts[peer],MPI_DOUBLE,
                                    peer,701,MPI_COMM_WORLD,&r);requests.push_back(r);
        }
        for(int peer=0;peer<size;++peer)if(sendcounts[peer]){
            MPI_Request r;MPI_Isend(outgoing.data()+senddisp[peer],sendcounts[peer],MPI_DOUBLE,
                                    peer,701,MPI_COMM_WORLD,&r);requests.push_back(r);
        }
    }
    void blocking(const std::vector<double>& x){
        pack(x);
        // Ordered nonblocking point-to-point calls followed by immediate wait:
        // baseline without compute/communication overlap.
        std::vector<MPI_Request> requests;post(requests);
        if(!requests.empty())MPI_Waitall((int)requests.size(),requests.data(),MPI_STATUSES_IGNORE);
    }

};
struct Timing {double halo=0,spmv=0,reduce=0,vector=0;};
struct RowGroups {std::vector<int> interior,boundary;};
static RowGroups classify_rows(const LocalCSR& a){
    RowGroups g;
    for(int i=0;i<a.end-a.begin;++i){
        bool remote=false;
        for(int j=a.rowptr[i];j<a.rowptr[i+1];++j)
            if(a.col[j]<a.begin || a.col[j]>=a.end){remote=true;break;}
        (remote?g.boundary:g.interior).push_back(i);
    }
    return g;
}

static void spmv(const LocalCSR& a,Halo& h,const std::vector<double>& x,
                 std::vector<double>& y,Timing& t,const std::string& comm,const RowGroups& groups){
    auto calculate=[&](int i){
        double sum=0;
        for(int j=a.rowptr[i];j<a.rowptr[i+1];++j){
            int c=a.col[j];
            double v=(c>=a.begin && c<a.end)?x[c-a.begin]:h.incoming[h.ghost_index.at(c)];
            sum+=a.val[j]*v;
        }
        y[i]=sum;
    };
    if(comm=="baseline"){
        double t0=MPI_Wtime();h.blocking(x);t.halo+=MPI_Wtime()-t0;
        t0=MPI_Wtime();
        #pragma omp parallel for schedule(static)
        for(int i=0;i<a.end-a.begin;++i)calculate(i);
        t.spmv+=MPI_Wtime()-t0;
    } else {
        double t0=MPI_Wtime();h.pack(x);
        std::vector<MPI_Request> requests;h.post(requests);
        t.halo+=MPI_Wtime()-t0;
        // Only rows with entirely local column indices are safe before Waitall.
        double tc=MPI_Wtime();
        #pragma omp parallel for schedule(static)
        for(int k=0;k<(int)groups.interior.size();++k)calculate(groups.interior[k]);
        t.spmv+=MPI_Wtime()-tc;
        t0=MPI_Wtime();
        if(!requests.empty())MPI_Waitall((int)requests.size(),requests.data(),MPI_STATUSES_IGNORE);
        t.halo+=MPI_Wtime()-t0;
        tc=MPI_Wtime();
        #pragma omp parallel for schedule(static)
        for(int k=0;k<(int)groups.boundary.size();++k)calculate(groups.boundary[k]);
        t.spmv+=MPI_Wtime()-tc;
    }
}
static double dot(const std::vector<double>& a,const std::vector<double>& b,Timing& t){
    double local=0;
    #pragma omp parallel for reduction(+:local) schedule(static)
    for(long long i=0;i<(long long)a.size();++i)local+=a[i]*b[i];
    double global=0;double t0=MPI_Wtime();
    MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    t.reduce+=MPI_Wtime()-t0;return global;
}
int main(int argc,char** argv){
    int provided=MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    if(provided<MPI_THREAD_FUNNELED){
        std::cerr<<"MPI does not provide MPI_THREAD_FUNNELED\n";
        MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
    int rank=0,size=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        int dim=2,m=16,maxiter=10000,fixed_iters=0;double tol=1e-8;std::string solution="smooth",comm="baseline";
        for(int i=1;i<argc;++i){
            std::string k=argv[i];if(i+1>=argc)throw std::runtime_error("missing value for "+k);
            std::string v=argv[++i];
            if(k=="--dim")dim=std::stoi(v);else if(k=="--grid")m=std::stoi(v);
            else if(k=="--tol")tol=std::stod(v);else if(k=="--max-iters")maxiter=std::stoi(v);
            else if(k=="--solution")solution=v;else if(k=="--comm")comm=v;else if(k=="--fixed-iters")fixed_iters=std::stoi(v);else throw std::runtime_error("unknown option "+k);
        }
        if((dim!=2 && dim!=3)||m<2||tol<=0||maxiter<1||fixed_iters<0||(comm!="baseline"&&comm!="overlap")||(solution!="ones"&&solution!="smooth"&&solution!="benchmark"))
            throw std::runtime_error("invalid arguments");
        long long n64=1;for(int d=0;d<dim;++d){if(n64>(std::numeric_limits<int>::max()-1LL)/m)throw std::runtime_error("matrix dimension overflow");n64*=m;}
        if(n64>std::numeric_limits<int>::max()-1||n64<size)
            throw std::runtime_error("matrix dimension must be >= MPI ranks and fit 32-bit indices");
        LocalCSR a=generate(dim,m,rank,size);Halo halo(a,rank,size);
        RowGroups groups=classify_rows(a);
        int nloc=a.end-a.begin;
        std::vector<double> truth(nloc),b(nloc),x(nloc,0),r(nloc),p(nloc),ap(nloc),ax(nloc);
        for(int i=0;i<nloc;++i)truth[i]=truth_at(a.begin+i,dim,m,solution);
        // RHS generation without an MPI exchange: x* is known analytically at every global index.
        for(int i=0;i<nloc;++i)
            for(int j=a.rowptr[i];j<a.rowptr[i+1];++j)
                b[i]+=a.val[j]*truth_at(a.col[j],dim,m,solution);
        r=b;p=r;
        const int omp_threads=omp_get_max_threads();
        Timing t;
        MPI_Barrier(MPI_COMM_WORLD);double t0=MPI_Wtime();
        double bb=dot(b,b,t),rr=bb;bool converged=(bb==0);int iterations=0;
        for(int k=0;k<(fixed_iters>0?fixed_iters:maxiter) && !converged;++k){
            spmv(a,halo,p,ap,t,comm,groups);
            double pap=dot(p,ap,t);
            if(!(pap>0)||!std::isfinite(pap))throw std::runtime_error("CG breakdown: nonpositive pAp");
            double alpha=rr/pap;
            double tv=MPI_Wtime();
            #pragma omp parallel for schedule(static)
            for(int i=0;i<nloc;++i){x[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
            t.vector+=MPI_Wtime()-tv;
            double rrnew=dot(r,r,t);iterations=k+1;
            if(!std::isfinite(rrnew))throw std::runtime_error("CG breakdown: nonfinite residual");
            if(std::sqrt(rrnew/bb)<tol){converged=true;break;}
            double beta=rrnew/rr;tv=MPI_Wtime();
            #pragma omp parallel for schedule(static)
            for(int i=0;i<nloc;++i)p[i]=r[i]+beta*p[i];
            t.vector+=MPI_Wtime()-tv;rr=rrnew;
        }
        double elapsed=MPI_Wtime()-t0;
        // Recompute TRUE residual, not just the CG recurrence residual.
        double validation_start=MPI_Wtime();
        Timing validation_timing;
        spmv(a,halo,x,ax,validation_timing,comm,groups);
        double errlocal=0,reslocal=0;
        #pragma omp parallel for reduction(+:errlocal,reslocal) schedule(static)
        for(int i=0;i<nloc;++i){
            double d=x[i]-truth[i],q=ax[i]-b[i];errlocal+=d*d;reslocal+=q*q;
        }
        double err=0,res=0,tnorm=0,localnorm=0;
        #pragma omp parallel for reduction(+:localnorm) schedule(static)
        for(int i=0;i<nloc;++i)localnorm+=truth[i]*truth[i];
        MPI_Allreduce(&errlocal,&err,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(&reslocal,&res,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(&localnorm,&tnorm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        double relerr=std::sqrt(err)/std::max(std::sqrt(tnorm),std::numeric_limits<double>::min());
        double relres=std::sqrt(res)/std::max(std::sqrt(bb),std::numeric_limits<double>::min());
        if(relres>=tol)converged=false;
        double validation_elapsed=MPI_Wtime()-validation_start;
        double maxelapsed=0,maxhalo=0,maxspmv=0,maxreduce=0,maxvector=0,maxvalidation=0;
        MPI_Reduce(&validation_elapsed,&maxvalidation,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&elapsed,&maxelapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t.halo,&maxhalo,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t.spmv,&maxspmv,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t.reduce,&maxreduce,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t.vector,&maxvector,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        long long nnzloc=(long long)a.val.size(),nnz=0;
        MPI_Reduce(&nnzloc,&nnz,1,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
        if(rank==0){
            std::cout<<std::setprecision(12)
                <<"dim,grid,n,nnz,solution,ranks,iterations,runtime_s,relative_residual,relative_error,converged,halo_s,spmv_s,allreduce_s,vector_s,comm,fixed_iters,validation_s,omp_threads\n"
                <<dim<<','<<m<<','<<a.n<<','<<nnz<<','<<solution<<','<<size<<','<<iterations<<','<<maxelapsed<<','
                <<relres<<','<<relerr<<','<<(converged?1:0)<<','<<maxhalo<<','<<maxspmv<<','<<maxreduce<<','<<maxvector<<','<<comm<<','<<fixed_iters<<','<<maxvalidation<<','<<omp_threads<<'\n';
        }
        MPI_Finalize();return converged?0:2;
    }catch(const std::exception& e){
        std::cerr<<"Rank "<<rank<<" ERROR: "<<e.what()<<'\n';
        MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
