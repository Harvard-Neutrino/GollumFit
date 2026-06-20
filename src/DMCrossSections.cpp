#include "DMCrossSections.h"
#include <cmath>
#include <stdexcept>
namespace gollumfit { namespace dm {

// DANSA's reference uses 80-digit Decimal. Several formulas have catastrophic
// cancellation (e.g. FV's 1 + mx^2/mp^2 - term2 with tiny mx), so we compute
// internally in long double (128-bit on Linux aarch64/x86_64) and return double.
using ld = long double;
static const ld PI = 3.14159265358979323846264338327950288L;

DMInteraction interaction_from_string(const std::string& n){
  if(n=="scalar")   return DMInteraction::scalar;
  if(n=="fermion")  return DMInteraction::fermion;
  if(n=="vector")   return DMInteraction::vector;
  if(n=="fermscal") return DMInteraction::fermscal;
  if(n=="sfermion") return DMInteraction::sfermion;
  throw std::runtime_error("unknown DM interaction: "+n);
}

static inline ld theta(ld Ei,ld Ef,ld mx){ return mx*(1.0L/Ef - 1.0L/Ei); }
static inline ld dthetadE(ld Ef,ld mx){ return mx/(Ef*Ef); }

static ld sigma_SS(ld Ei,ld g,ld mp,ld mx){
  const ld g2=g*g, Ei2=Ei*Ei, mp2=mp*mp, mx2=mx*mx;
  ld t1=4.0L*Ei2*mx, t2=2.0L*Ei*mp2+4.0L*Ei2*mx+mp2*mx;
  ld coeff=-g2/(64.0L*PI*Ei2*mx2);
  return coeff*(t1/t2 + std::log(mp2*(2.0L*Ei+mx)) - std::log(t2));
}
static ld sigma_FS(ld Ei,ld g,ld mp,ld mx){
  const ld g2=g*g, Ei2=Ei*Ei, mp2=mp*mp, mx2=mx*mx;
  ld coeff=g2/(32.0L*PI*Ei2*mx2);
  ld t1=Ei*mx;
  ld t2=Ei*mx2/(2.0L*Ei+mx);
  ld t3=(Ei*mx2*mp2*(mp2-4.0L*mx2))/((2.0L*Ei*mx+mp2)*(4.0L*Ei2*mx+2.0L*Ei*mp2+mx*mp2));
  ld t4=(Ei*mx*(mp2-4.0L*mx2))/(2.0L*Ei*mx+mp2);
  ld t5=(mp2-2.0L*mx2)*std::log(mp2*(2.0L*Ei+mx)/(4.0L*Ei2*mx+2.0L*Ei*mp2+mx*mp2));
  return coeff*(t1-t2-t3+t4+t5);
}
static ld sigma_FV(ld Ei,ld g,ld mp,ld mx){
  const ld g2=g*g, Ei2=Ei*Ei, mp2=mp*mp, mx2=mx*mx;
  ld coeff=g2/(16.0L*PI*Ei2*mx2);
  ld t1=mp2+mx2+2.0L*Ei*mx;
  ld logs=std::log(mp2*(2.0L*Ei+mx)/(mx*(4.0L*Ei2+mp2)+2.0L*Ei*mp2));
  ld t2=(2.0L*Ei*(4.0L*Ei2*mx+Ei*(mx2+2.0L*mp2)+mx*mp2))/((2.0L*Ei+mx)*(mx*(4.0L*Ei2+mp2)+2.0L*Ei*mp2));
  ld t3=4.0L*Ei2*(1.0L+mx2/mp2-t2);
  return coeff*(t1*logs+t3);
}
static ld sigma_SF(ld Ei,ld g,ld mp,ld mx){
  if(mx>mp) return 0.0L;
  const ld g2=g*g, Ei2=Ei*Ei, mp2=mp*mp, mx2=mx*mx;
  ld coeff=g2/(64.0L*PI);
  ld t1=8.0L*Ei2*mx/((2.0L*Ei+mx)*std::pow(mp2-mx*(2.0L*Ei+mx),2));
  ld t2=4.0L/(-2.0L*Ei*mx+mx2-mp2);
  ld t3=8.0L/(mx*(2.0L*Ei+mx)-mp2);
  ld t4=(4.0L*Ei*mx-2.0L*(mx2+3.0L*mp2))/(Ei*mx*(mx*(2.0L*Ei+mx)-mp2)) + 3.0L/Ei2;
  ld t5=4.0L*Ei2*mx/(mp2*(2.0L*Ei+mx)-mx*mx*mx)+1.0L;
  if(t5<=0.0L) return 0.0L;
  return coeff*(t1+t2+t3+t4*std::log(t5));
}
static ld sigma_FSres(ld Ei,ld g,ld mp,ld mx){
  if(mx>=mp) return 0.0L;
  const ld g2=g*g, Ei2=Ei*Ei, mp2=mp*mp, mx2=mx*mx, mx3=mx*mx*mx, mp4=mp2*mp2;
  ld coeff=g2/(16.0L*PI*Ei2*mx2);
  ld t0=2.0L*mx3*Ei2*(2.0L*Ei+mx)/std::pow(mx*(2.0L*Ei+mx)-mp2,2);
  ld t1=Ei*mx;
  ld t2=Ei*mx2/(2.0L*Ei+mx);
  ld t3=Ei*mx*mp4/((mp2-mx2)*(mp2+mx*(2.0L*Ei-mx)));
  ld t4=Ei*mx2*mp4/((mp2-mx2)*(mp2*(2.0L*Ei+mx)-mx3));
  ld t5=(mp2*(2.0L*Ei+mx)-mx3)/((2.0L*Ei+mx)*(mp2+mx*(2.0L*Ei-mx)));
  return coeff*(t0+t1+t2+t3+t4+mp2*std::log(t5));
}

double sigma(DMInteraction in,double Ei,double g,double mp,double mx){
  switch(in){
    case DMInteraction::scalar:   return (double)sigma_SS(Ei,g,mp,mx);
    case DMInteraction::fermion:  return (double)sigma_FS(Ei,g,mp,mx);
    case DMInteraction::vector:   return (double)sigma_FV(Ei,g,mp,mx);
    case DMInteraction::fermscal: return (double)sigma_SF(Ei,g,mp,mx);
    case DMInteraction::sfermion: return (double)sigma_FSres(Ei,g,mp,mx);
  }
  return 0.0;
}

static ld dxs_SS(ld Ei,ld Ef,ld g,ld mp,ld mx){
  ld th=theta(Ei,Ef,mx); const ld g2=g*g,Ei2=Ei*Ei,mp2=mp*mp;
  return dthetadE(Ef,mx)*g2/(16.0L*PI)*(th*Ei2*mx)/((th*Ei+mx)*std::pow(th*Ei*mp2+mx*(mp2+2.0L*th*Ei2),2));
}
static ld dxs_FS(ld Ei,ld Ef,ld g,ld mp,ld mx){
  ld th=-theta(Ei,Ef,mx); const ld g2=g*g,Ei2=Ei*Ei,mp2=mp*mp,mx2=mx*mx;
  return dthetadE(Ef,mx)*g2*th*Ei2*mx2*(2.0L*th*Ei*mx+th*Ei2-2.0L*mx2)
         /(8.0L*PI*std::pow(mx-th*Ei,2)*std::pow(th*Ei*mp2-mx*(mx2-2.0L*th*Ei2),2));
}
static ld dxs_FV(ld Ei,ld Ef,ld g,ld mp,ld mx){
  ld th=theta(Ei,Ef,mx); const ld g2=g*g,Ei2=Ei*Ei,mp2=mp*mp,mx2=mx*mx;
  return dthetadE(Ef,mx)*g2*Ei2*mx2*(th*(2.0L-th)*Ei*mx+th*th*Ei2+(2.0L-th)*mx2)
         /(4.0L*PI*std::pow(th*Ei+mx,2)*std::pow(th*Ei*mp2+mx*(mp2+2.0L*th*Ei2),2));
}
static ld dxs_SF(ld Ei,ld Ef,ld g,ld mp,ld mx){
  ld th=theta(Ei,Ef,mx); const ld g2=g*g,mp2=mp*mp,mx2=mx*mx;
  ld Ei4=Ei*Ei*Ei*Ei, mx5=mx*mx*mx*mx*mx;
  return dthetadE(Ef,mx)*g2*(2.0L-th)*Ei4*mx5*std::pow(th*Ei+2.0L*mx,2)
         /((4.0L*PI)*std::pow(mp2-mx*(2.0L*Ei+mx),2)*std::pow(th*Ei+mx,3)
           *std::pow(mx*mx*mx-Ei*(th*mp2+(2.0L-th)*mx2)-mx*mp2,2));
}
static ld dxs_FSres(ld Ei,ld Ef,ld g,ld mp,ld mx){
  ld th=theta(Ei,Ef,mx); const ld g2=g*g,mp2=mp*mp,mx2=mx*mx;
  return dthetadE(Ef,mx)*g2*mx2*std::pow(2.0L*Ei+mx,2)
         /(16.0L*PI*std::pow(th*Ei+mx,2)*std::pow(mx*(2.0L*Ei+mx)-mp2,2));
}

double dsigmade(DMInteraction in,double Ei,double Ef,double g,double mp,double mx){
  if(Ei<=Ef) return 0.0;
  switch(in){
    case DMInteraction::scalar:   return (double)dxs_SS(Ei,Ef,g,mp,mx);
    case DMInteraction::fermion:  return (double)dxs_FS(Ei,Ef,g,mp,mx);
    case DMInteraction::vector:   return (double)dxs_FV(Ei,Ef,g,mp,mx);
    case DMInteraction::fermscal: return (double)dxs_SF(Ei,Ef,g,mp,mx);
    case DMInteraction::sfermion: return (double)dxs_FSres(Ei,Ef,g,mp,mx);
  }
  return 0.0;
}

}} // namespace
