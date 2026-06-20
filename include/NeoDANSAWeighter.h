#ifndef NEODANSAWEIGHTER_H_
#define NEODANSAWEIGHTER_H_
#include <cmath>
#include "Event.h"
#include "DMAttenuation.h"

namespace gollumfit {

// NeoDANSA per-event astro+galactic signal weight (DM-attenuated), mirroring
// dansa.analysis.events.AstroEvents. Energies in GeV; exposure in seconds.
// component: 0 = astro+galactic, 1 = astro only, 2 = galactic only.
struct NeoDANSAWeighter {
  static constexpr double C0 = 1.0e-18;                 // GeV^-1 cm^-2 s^-1 sr^-1
  static constexpr double PIVOT_GEV = 1.0e5;            // 100 TeV

  const dm::DMAttenuator* attAstro = nullptr;   // prepared with gammaAstro
  const dm::DMAttenuator* attGal = nullptr;     // prepared with gammaGalactic
  double astroNorm = 0.0, normGalactic = 0.0;
  double gammaAstro = 2.53, gammaGalactic = 2.7;
  double exposure = 304100756.4628376;          // s; set to the analysis exposure (burn sample = x0.1)
  int component = 0;

  double operator()(const Event& e) const {
    double w = 0.0;
    if(component==0 || component==1){
      double att = attAstro->att_one(e.columnDens, e.primaryEnergy);
      w += astroNorm * C0 * std::pow(e.primaryEnergy/PIVOT_GEV, -gammaAstro)
           * exposure * e.oneWeight * att;
    }
    if(component==0 || component==2){
      double att = attGal->att_one(e.columnDensGalactic, e.primaryEnergy);
      w += normGalactic * C0 * std::pow(e.primaryEnergy/PIVOT_GEV, -gammaGalactic)
           * exposure * e.oneWeight * e.spatialTemplate * att;
    }
    return w;
  }
};

} // namespace gollumfit
#endif
