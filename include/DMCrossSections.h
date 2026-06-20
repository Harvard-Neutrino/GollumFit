#ifndef DMCROSSSECTIONS_H_
#define DMCROSSSECTIONS_H_
#include <string>
namespace gollumfit { namespace dm {

enum class DMInteraction { scalar, fermion, vector, fermscal, sfermion };
DMInteraction interaction_from_string(const std::string& name);

// Ei, Ef, mphi, mx in eV. g is the DANSA coupling (g_chi*g_nu).
double sigma(DMInteraction in, double Ei, double g, double mphi, double mx);
double dsigmade(DMInteraction in, double Ei, double Ef, double g, double mphi, double mx);

}} // namespace
#endif
