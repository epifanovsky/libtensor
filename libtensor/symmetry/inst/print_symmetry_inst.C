#include <libtensor/core/scalar_transf_double.h>
#include "../print_symmetry.h"

namespace libtensor {


template<>
std::ostream &operator<<(std::ostream &os, const scalar_transf<double> &tr) {

    os << tr.get_coeff();
    return os;
}


template<>
std::ostream &operator<<(std::ostream &os, const scalar_transf<float> &tr) {

    os << tr.get_coeff();
    return os;
}



} // namespace libtensor



