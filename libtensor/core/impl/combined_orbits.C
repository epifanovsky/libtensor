#include <libtensor/core/scalar_transf_double.h>
#include "combined_orbits_impl.h"

namespace libtensor {


template class combined_orbits<1, double>;
template class combined_orbits<2, double>;
template class combined_orbits<3, double>;
template class combined_orbits<4, double>;
template class combined_orbits<5, double>;
template class combined_orbits<6, double>;
template class combined_orbits<7, double>;
template class combined_orbits<8, double>;

template class combined_orbits<1, float>;
template class combined_orbits<2, float>;
template class combined_orbits<3, float>;
template class combined_orbits<4, float>;
template class combined_orbits<5, float>;
template class combined_orbits<6, float>;
template class combined_orbits<7, float>;
template class combined_orbits<8, float>;

} // namespace libtensor
