#include "tensor_ctrl.h"
#include "tod_set.h"

namespace libtensor {

void tod_set::perform(tensor_i<double> &t) throw(exception) {
	tensor_ctrl<double> tctrl(t);
	double *d = tctrl.req_dataptr();
	size_t sz = t.get_dims().get_size();
	#pragma unroll(8)
	for(size_t i=0; i<sz; i++) d[i] = m_val;
	tctrl.ret_dataptr(d);
}

} // namespace libtensor

