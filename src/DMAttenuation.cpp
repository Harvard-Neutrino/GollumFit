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
    double lg = (LOGEMIN-1.0) + ((LOGEMAX+1.0)-(LOGEMIN-1.0))*k/(NN-1); // logspace(2,7,150) GeV
    nodes_[k] = std::pow(10.0, lg) * GeV;                               // eV
  }
  lognode_.resize(NN);
  for(int k=0;k<NN;++k) lognode_[k] = std::log10(nodes_[k]);
  build_eigsystem();
}

void DMAttenuator::build_eigsystem(){
  std::vector<std::vector<double>> M(NN, std::vector<double>(NN, 0.0));
  for(int k=0;k<NN;++k) M[k][k] = -sigma(in_, nodes_[k], g_, mphi_, mx_);
  std::vector<double> dlog(NN-1);
  for(int k=0;k<NN-1;++k) dlog[k] = std::log(nodes_[k+1]) - std::log(nodes_[k]);
  for(int i=0;i<NN;++i)
    for(int j=i+1;j<NN;++j)
      M[i][j] += dlog[j-1] * dsigmade(in_, nodes_[j], nodes_[i], g_, mphi_, mx_) / nodes_[j];

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

void DMAttenuator::prepare(double gamma) const {
  if(prepared_gamma_ == gamma && !ci_.empty()) return;
  // phi0 = nodes^(-gamma); solve v * ci = phi0 via SVD least squares (like np.linalg.lstsq).
  gsl_matrix* U = gsl_matrix_alloc(NN,NN);
  for(int i=0;i<NN;++i) for(int j=0;j<NN;++j) gsl_matrix_set(U,i,j,v_[i][j]);
  gsl_matrix* Vv=gsl_matrix_alloc(NN,NN);
  gsl_vector* S=gsl_vector_alloc(NN); gsl_vector* work=gsl_vector_alloc(NN);
  gsl_linalg_SV_decomp(U,Vv,S,work);
  gsl_vector* phi0=gsl_vector_alloc(NN); gsl_vector* ci=gsl_vector_alloc(NN);
  for(int k=0;k<NN;++k) gsl_vector_set(phi0,k,std::pow(nodes_[k],-gamma));
  gsl_linalg_SV_solve(U,Vv,S,phi0,ci);
  ci_.resize(NN); phi0_.resize(NN);
  for(int k=0;k<NN;++k){ ci_[k]=gsl_vector_get(ci,k); phi0_[k]=std::pow(nodes_[k],-gamma); }
  gsl_vector_free(phi0); gsl_vector_free(ci); gsl_vector_free(S); gsl_vector_free(work);
  gsl_matrix_free(U); gsl_matrix_free(Vv);
  prepared_gamma_ = gamma;
}

double DMAttenuator::att_one(double columnDens, double trueE_gev) const {
  if(g_==0.0) return 1.0;
  const double thresh = LOGEMAX + std::log10(GeV);
  double logE = std::log10(trueE_gev*GeV);
  if(logE>thresh) return 1.0;
  double t = columnDens*(GeV/(CM*CM))/mx_;
  // bracket j such that lognode_[j] <= logE < lognode_[j+1]
  int j=0; while(j<NN-2 && lognode_[j+1]<logE) ++j;
  // phisol only at the two bracketing nodes (avoids O(N^2) per event)
  double accj=0.0, accj1=0.0;
  for(int k=0;k<NN;++k){
    double ek = ci_[k]*std::exp(w_[k]*t);
    accj  += v_[j][k]   * ek;
    accj1 += v_[j+1][k] * ek;
  }
  double phij  = accj  / phi0_[j];
  double phij1 = accj1 / phi0_[j+1];
  double d=(logE-lognode_[j])/(lognode_[j+1]-lognode_[j]);
  return (1.0-d)*phij + d*phij1;
}

std::vector<double> DMAttenuator::attenuation(double gamma,
    const std::vector<double>& cd, const std::vector<double>& trueE) const {
  prepare(gamma);
  std::vector<double> out(trueE.size(), 1.0);
  for(size_t e=0;e<trueE.size();++e) out[e]=att_one(cd[e], trueE[e]);
  return out;
}

}} // namespace
