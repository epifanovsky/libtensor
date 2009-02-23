#include "tod_set.h"

namespace libtensor {

void tod_set::perform(tensor_i<double> &t) throw(exception) {
	double *d = req_dataptr(t, req_simplest_permutation(t));
	size_t sz = t.get_dims().get_size();
	#pragma unroll(8)
	for(size_t i=0; i<sz; i++) d[i] = m_val;
	ret_dataptr(t, d);
}

} // namespace libtensor

