#include "DMAttenuation.h"
#include <cmath>
#include <stdexcept>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>

namespace gollumfit { namespace dm {

static constexpr int NN = 150;
static constexpr double LOGEMIN = 3.0, LOGEMAX = 6.0, GeV = 1.0e9;
// cm in natural units: cm = 1e-2 * meter, meter = 5.06773093741e6 (1/eV)
static constexpr double CM = 1.0e-2 * 5.06773093741e6;

DMAttenuator::DMAttenuator(DMInteraction in,double g,double mphi,double mx)
  : in_(in), g_(g), mphi_(mphi), mx_(mx) {
  nodes_.resize(NN);
  for(int k=0;k<NN;++k){
    // logspace(logemin-1, logemax+1, 150) in GeV -> eV
    double lg = (LOGEMIN-1.0) + ((LOGEMAX+1.0)-(LOGEMIN-1.0))*k/(NN-1);
    nodes_[k] = std::pow(10.0, lg) * GeV;
  }
  build_eigsystem();
}

void DMAttenuator::build_eigsystem(){
  // Cascade matrix M = -Diag(sigma) + RHS (upper triangular).
  std::vector<std::vector<double>> M(NN, std::vector<double>(NN, 0.0));
  for(int k=0;k<NN;++k) M[k][k] = -sigma(in_, nodes_[k], g_, mphi_, mx_);
  std::vector<double> dlog(NN-1);
  for(int k=0;k<NN-1;++k) dlog[k] = std::log(nodes_[k+1]) - std::log(nodes_[k]);
  for(int i=0;i<NN;++i)
    for(int j=i+1;j<NN;++j)
      M[i][j] += dlog[j-1] * dsigmade(in_, nodes_[j], nodes_[i], g_, mphi_, mx_) / nodes_[j];

  // Eigen-decomposition (general real matrix). Use GSL's paired (eval, evec)
  // taking real parts -- consistent pairing, matching DANSA's numpy.linalg.eig.
  gsl_matrix* A = gsl_matrix_alloc(NN,NN);
  for(int i=0;i<NN;++i) for(int j=0;j<NN;++j) gsl_matrix_set(A,i,j,M[i][j]);
  gsl_vector_complex* eval = gsl_vector_complex_alloc(NN);
  gsl_matrix_complex* evec = gsl_matrix_complex_alloc(NN,NN);
  gsl_eigen_nonsymmv_workspace* ws = gsl_eigen_nonsymmv_alloc(NN);
  gsl_eigen_nonsymmv(A, eval, evec, ws);

  w_.resize(NN);
  v_.assign(NN, std::vector<double>(NN, 0.0));
  for(int k=0;k<NN;++k){
    w_[k] = GSL_REAL(gsl_vector_complex_get(eval,k));
    for(int j=0;j<NN;++j) v_[j][k] = GSL_REAL(gsl_matrix_complex_get(evec,j,k));
  }
  gsl_eigen_nonsymmv_free(ws); gsl_matrix_complex_free(evec);
  gsl_vector_complex_free(eval); gsl_matrix_free(A);
}

std::vector<double> DMAttenuator::attenuation(double gamma,
    const std::vector<double>& cd, const std::vector<double>& trueE) const {
  std::vector<double> out(trueE.size(), 1.0);
  if(g_==0.0) return out;

  // phi0 = nodes^(-gamma); solve v * ci = phi0 (least squares via SVD, like np.linalg.lstsq)
  gsl_matrix* U = gsl_matrix_alloc(NN,NN);
  for(int i=0;i<NN;++i) for(int j=0;j<NN;++j) gsl_matrix_set(U,i,j,v_[i][j]);
  gsl_matrix* Vv=gsl_matrix_alloc(NN,NN);
  gsl_vector* S=gsl_vector_alloc(NN); gsl_vector* work=gsl_vector_alloc(NN);
  gsl_linalg_SV_decomp(U,Vv,S,work);
  gsl_vector* phi0=gsl_vector_alloc(NN); gsl_vector* ci=gsl_vector_alloc(NN);
  for(int k=0;k<NN;++k) gsl_vector_set(phi0,k,std::pow(nodes_[k],-gamma));
  gsl_linalg_SV_solve(U,Vv,S,phi0,ci);
  std::vector<double> civ(NN), phi0v(NN);
  for(int k=0;k<NN;++k){ civ[k]=gsl_vector_get(ci,k); phi0v[k]=std::pow(nodes_[k],-gamma); }

  std::vector<double> lognode(NN); for(int k=0;k<NN;++k) lognode[k]=std::log10(nodes_[k]);
  const double thresh = LOGEMAX + std::log10(GeV);
  for(size_t e=0;e<trueE.size();++e){
    double E = trueE[e]*GeV;             // eV
    double logE = std::log10(E);
    if(logE>thresh) { out[e]=1.0; continue; }
    double t = cd[e]*(GeV/(CM*CM))/mx_;
    std::vector<double> phisol(NN);
    for(int j=0;j<NN;++j){
      double acc=0.0;
      for(int k=0;k<NN;++k) acc += v_[j][k]*civ[k]*std::exp(w_[k]*t);
      phisol[j]=acc/phi0v[j];
    }
    int j=0; while(j<NN-2 && lognode[j+1]<logE) ++j;
    double d=(logE-lognode[j])/(lognode[j+1]-lognode[j]);
    out[e]=(1.0-d)*phisol[j]+d*phisol[j+1];
  }
  gsl_vector_free(phi0); gsl_vector_free(ci); gsl_vector_free(S); gsl_vector_free(work);
  gsl_matrix_free(U); gsl_matrix_free(Vv);
  return out;
}

}} // namespace
