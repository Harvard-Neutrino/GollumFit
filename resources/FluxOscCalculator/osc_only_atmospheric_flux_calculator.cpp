// NeoDANSA: propagate an atmospheric surface flux with OSCILLATIONS ONLY (no Earth
// interactions/absorption) over the FULL zenith range, writing a nuSQuIDS-format HDF5.
//
// Rationale: NuGen MC already accounts for Earth absorption, so the flux table must NOT
// include it (else double-counted). nuSQUIDSAtm with iinteraction=false keeps the coherent
// matter potential (MSW oscillations) but drops the non-coherent absorption/NC/tau terms.
//
// Input .dat: 8 columns per (cos(theta), E) row, cos-major:
//   cosTheta  E[GeV]  nue  nuebar  numu  numubar  nutau  nutaubar   (surface flux dN/dE)
// Grid MUST match the constructor below: linspace(-1,1,100) x logspace(1e2,1e7,350) GeV.
//
// Usage: osc_only_atmospheric_flux_calculator  influx.dat  EARTH_MODEL_PREM.dat  out.hdf5
#include <vector>
#include <iostream>
#include <string>
#include <cassert>
#include <nuSQuIDS/nuSQuIDS.h>
#include <nuSQuIDS/marray.h>

using namespace nusquids;

int main(int argc, char* argv[]){
  if(argc != 4){
    printf("USAGE: %s influx.dat earth_model.dat out.hdf5\n", argv[0]);
    return 1;
  }
  std::string input_flux_path = argv[1];
  std::string input_earth_path = argv[2];
  std::string output_path = argv[3];

  const unsigned int numneu = 3;
  const squids::Const units;

  // iinteraction = FALSE -> oscillations only (matter potential kept, absorption/NC off).
  nuSQUIDSAtm<> nus_atm(linspace(-1., 1., 100),
                        logspace(1.e2*units.GeV, 1.e7*units.GeV, 350),
                        numneu, both, /*iinteraction=*/false);

  std::shared_ptr<EarthAtm> earth = std::make_shared<EarthAtm>(input_earth_path);
  nus_atm.Set_EarthModel(earth);
  nus_atm.Set_ProgressBar(false);

  // NuFit 3-flavor oscillation parameters (same central values as the shipped calculators).
  nus_atm.Set_MixingAngle(0, 1, 0.585732);
  nus_atm.Set_MixingAngle(0, 2, 0.147655);
  nus_atm.Set_MixingAngle(1, 2, 0.726057);
  nus_atm.Set_SquareMassDifference(1, 7.50e-05);
  nus_atm.Set_SquareMassDifference(2, 0.00252);
  nus_atm.Set_CPPhase(0, 2, 0.0);

  double error = 1.0e-15;
  nus_atm.Set_GSL_step(gsl_odeiv2_step_rk4);
  nus_atm.Set_rel_error(error);
  nus_atm.Set_abs_error(error);

  marray<double,2> input_flux = quickread(input_flux_path);
  marray<double,4> inistate {nus_atm.GetNumCos(), nus_atm.GetNumE(), 2, numneu};
  std::fill(inistate.begin(), inistate.end(), 0);

  marray<double,1> cos_range = nus_atm.GetCosthRange();
  marray<double,1> e_range = nus_atm.GetERange();
  assert( input_flux.extent(0) == nus_atm.GetNumCos()*nus_atm.GetNumE() );

  for(int ci = 0; ci < (int)nus_atm.GetNumCos(); ci++){
    for(int ei = 0; ei < (int)nus_atm.GetNumE(); ei++){
      double enu = e_range[ei]/units.GeV;
      double cth = cos_range[ci];
      size_t r = ci*e_range.size() + ei;
      assert( std::fabs(enu - input_flux[r][1]) < 1.e-3*enu );
      assert( std::fabs(cth - input_flux[r][0]) < 1.e-4 );
      inistate[ci][ei][0][0] = input_flux[r][2];  // nue
      inistate[ci][ei][0][1] = input_flux[r][4];  // numu
      inistate[ci][ei][0][2] = input_flux[r][6];  // nutau
      inistate[ci][ei][1][0] = input_flux[r][3];  // nuebar
      inistate[ci][ei][1][1] = input_flux[r][5];  // numubar
      inistate[ci][ei][1][2] = input_flux[r][7];  // nutaubar
    }
  }

  nus_atm.Set_initial_state(inistate, flavor);
  nus_atm.EvolveState();
  nus_atm.WriteStateHDF5(output_path);
  std::cout << "wrote " << output_path << std::endl;
  return 0;
}
