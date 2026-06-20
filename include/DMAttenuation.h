#ifndef DMATTENUATION_H_
#define DMATTENUATION_H_
#include <vector>
#include <string>
#include <cmath>
#include "DMCrossSections.h"
namespace gollumfit { namespace dm {

// Solves the DM cascade equation on a 150-node grid and returns the per-event
// neutrino survival probability, matching DANSA's SurvivalProbability.
//
// Usage in a per-event weighter: construct once on (interaction,g,mphi,mx)
// (builds the eigensystem), call prepare(gamma) once, then att_one(cd,trueE)
// per event (cheap: only the two bracketing energy nodes are evaluated).
class DMAttenuator {
 public:
  DMAttenuator(DMInteraction in, double g, double mphi, double mx); // builds nodes + eigensystem

  // Compute + cache ci for this gamma (re-call with a new gamma recomputes).
  void prepare(double gamma) const;
  // Per-event survival probability using the eigensystem + the prepared ci.
  double att_one(double columnDens, double trueE_gev) const;

  // Convenience: prepare(gamma) then att_one over all events (Plan-3 parity API).
  std::vector<double> attenuation(double gamma,
                                  const std::vector<double>& columnDens,
                                  const std::vector<double>& trueE_gev) const;
 private:
  DMInteraction in_; double g_, mphi_, mx_;
  std::vector<double> nodes_;            // eV, size 150
  std::vector<double> w_;                // eigenvalues (real part), paired with v_ columns
  std::vector<std::vector<double>> v_;   // eigenvectors, v_[j][k] (real part)
  std::vector<double> lognode_;          // log10(nodes_)
  void build_eigsystem();
  // prepared state (mutable: set by prepare(), used by const att_one)
  mutable double prepared_gamma_ = std::nan("");
  mutable std::vector<double> ci_;       // solution of v * ci = nodes^(-gamma)
  mutable std::vector<double> phi0_;     // nodes^(-gamma)
};

}} // namespace
#endif
