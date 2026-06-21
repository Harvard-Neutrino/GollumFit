#ifndef NEODANSAWEIGHTER_H_
#define NEODANSAWEIGHTER_H_
#include <cmath>
#include "Event.h"
#include "DMAttenuation.h"
#include "analysisWeighting.h"   // gollumfit::ConvFluxWeigther, gollumfit::brokenpowerlawTiltWeighter

namespace gollumfit {

// NeoDANSA per-event weight built on GollumFit's NATIVE weighter structs:
//   astro  = astroNorm * cachedAstroWeight * brokenpowerlawTilt(pivot,dGamma,dGamma2) * DMatt
//   galactic = normGalactic * cachedGalacticWeight * spatialTemplate * DMatt
//   conv   = convNorm * ConvFluxWeigther(16 DAEMONFlux params)   [cachedConvWeight + 16 gradient caches]
//   muon   = muonNorm * cachedMuonWeight
// cachedAstroWeight/cachedConvWeight/gradient caches are built in the loader from
// GollumFit's shipped nuSQuIDS flux + gradient tables x OneWeight (so GollumFit
// computes the gradients). The DM cascade attenuation is the only bespoke physics.
// component: 0=total, 1=astro, 2=galactic, 3=atmo-nu(conv), 4=muon.
struct NeoDANSAWeighter {
  const dm::DMAttenuator* attAstro = nullptr;   // prepared with the astro index
  const dm::DMAttenuator* attGal = nullptr;     // prepared with gammaGalactic

  double astroNorm = 0.0, normGalactic = 0.0, convNorm = 1.0, muonNorm = 1.0;
  double astroPivot = 5.0;          // log10(E_pivot / GeV); break energy of the tilt
  double astroDeltaGamma = 0.0;     // index tilt below pivot (DANSA single PL: == astroDeltaGammaSec)
  double astroDeltaGammaSec = 0.0;  // index tilt above pivot (broken-PL superset)
  // 16 DAEMONFlux conventional-flux nuisances (default 0 = nominal)
  double hekp = 0, hekm = 0, vhe1pip = 0, vhe1pim = 0, vhe3kp = 0, vhe3km = 0,
         vhe3pip = 0, vhe3pim = 0, vhe3p = 0, vhe3n = 0,
         cr1 = 0, cr2 = 0, cr3 = 0, cr4 = 0, cr5 = 0, cr6 = 0;
  int component = 0;

  double operator()(const Event& e) const {
    double w = 0.0;
    if((component==0 || component==1) && e.cachedAstroWeight != 0.0){
      double tilt = brokenpowerlawTiltWeighter<Event,double>(astroPivot, astroDeltaGamma,
                                                             astroDeltaGammaSec)(e);
      double att = attAstro->att_one(e.columnDens, e.primaryEnergy);
      w += astroNorm * e.cachedAstroWeight * tilt * att;
    }
    if((component==0 || component==2) && e.cachedGalacticWeight != 0.0){
      double att = attGal->att_one(e.columnDensGalactic, e.primaryEnergy);
      w += normGalactic * e.cachedGalacticWeight * e.spatialTemplate * att;
    }
    if(component==0 || component==3){
      // GollumFit's native conventional-flux weighter: cachedConvWeight + sum nuisance*gradient.
      ConvFluxWeigther<Event,double> cfw(hekp, hekm, vhe1pip, vhe1pim, vhe3kp, vhe3km,
                                         vhe3pip, vhe3pim, vhe3p, vhe3n,
                                         cr1, cr2, cr3, cr4, cr5, cr6);
      w += convNorm * cfw(e);   // returns 0 for signal events (cachedConvWeight + caches == 0)
    }
    if(component==0 || component==4) w += muonNorm * e.cachedMuonWeight;
    return w;
  }
};

} // namespace gollumfit
#endif
