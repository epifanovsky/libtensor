#include "symmetry.h"

namespace libtensor {

symmetry::symmetry(const symmetry_i &s, const dimensions &d) {
}

symmetry::~symmetry() {
}

const index &symmetry::get_unique(const index &i) const throw(exception) {
}

const permutation &symmetry::get_perm(const index &i) const throw(exception) {
}

double symmetry::get_coeff(const index &i) const throw(exception) {
}

} // namespace libtensor

