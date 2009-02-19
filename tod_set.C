#include "tod_set.h"

namespace libtensor {

void tod_set::perform(tensor_i<double> &t) throw(exception) {
	// Choose the easiest permutation first
	permutation p(t.get_dims().get_order());
	double *d = req_dataptr(t, p);
	size_t sz = t.get_dims().get_size();
	for(size_t i=0; i<sz; i++) d[i] = m_val;
	ret_dataptr(t, d);
}

} // namespace libtensor

