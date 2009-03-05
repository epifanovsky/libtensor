#include "tod_contract2.h"

namespace libtensor {

tod_contract2::tod_contract2(const size_t n, tensor_i<double> &t1,
	const permutation &p1, tensor_i<double> &t2, const permutation &p2,
	const permutation &pres) throw(exception) {
}

tod_contract2::~tod_contract2() {
}

void tod_contract2::prefetch() throw(exception) {
}

void tod_contract2::perform(tensor_i<double> &t) throw(exception) {
}

void tod_contract2::perform(tensor_i<double> &t, const double c)
	throw(exception) {
}

} // namespace libtensor

