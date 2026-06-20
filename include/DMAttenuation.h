#ifndef DMATTENUATION_H_
#define DMATTENUATION_H_
#include <vector>
#include <string>
#include "DMCrossSections.h"
namespace gollumfit { namespace dm {

// Solves the DM cascade equation on a 150-node grid and returns the per-event
// neutrino survival probability, matching DANSA's SurvivalProbability.
class DMAttenuator {
 public:
  DMAttenuator(DMInteraction in, double g, double mphi, double mx); // builds nodes + eigensystem
  // Per-event survival probability for each (columnDens[i], trueE_gev[i]).
  std::vector<double> attenuation(double gamma,
                                  const std::vector<double>& columnDens,
                                  const std::vector<double>& trueE_gev) const;
 private:
  DMInteraction in_; double g_, mphi_, mx_;
  std::vector<double> nodes_;            // eV, size 150
  std::vector<double> w_;                // eigenvalues (real part), paired with v_ columns
  std::vector<std::vector<double>> v_;   // eigenvectors, v_[j][k] (real part)
  void build_eigsystem();
};

}} // namespace
#endif
