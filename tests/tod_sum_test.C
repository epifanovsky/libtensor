#include "tod_sum_test.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

typedef tensor<double, libvmm::std_allocator<double> > tensor_d;

void tod_sum_test::perform() throw(libtest::test_exception) {
	index i1(4), i2(4);
	i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
	index_range ir(i1, i2);
	dimensions dim(ir);
	tensor_d t(dim);

	testop_set setop;
	tod_sum op(setop);
	op.perform(t);

	tensor_ctrl<double> tctrl(t);
	const double *p = tctrl.req_const_dataptr();
	bool ok = true;
	size_t sz = dim.get_size();
	for(size_t i=0; i<sz; i++) if(p[i]!=(double)i) { ok = false; break; }
	tctrl.ret_dataptr(p);

	if(!ok) {
		fail_test("tod_sum_test::perform()", __FILE__, __LINE__,
			"Set operation failed");
	}
}

void tod_sum_test::testop_set::perform(tensor_i<double> &t) throw(exception) {
	size_t sz = t.get_dims().get_size();
	tensor_ctrl<double> tctrl(t);
	double *p = tctrl.req_dataptr();
	for(size_t i=0; i<sz; i++) p[i] = (double)i;
	tctrl.ret_dataptr(p);
}

void tod_sum_test::testop_add::perform(tensor_i<double> &t) throw(exception) {
	size_t sz = t.get_dims().get_size();
	tensor_ctrl<double> tctrl(t);
	double *p = tctrl.req_dataptr();
	for(size_t i=0; i<sz; i++) p[i] += m_v;
	tctrl.ret_dataptr(p);
}

void tod_sum_test::testop_add::perform(tensor_i<double> &t, double c)
	throw(exception) {
	size_t sz = t.get_dims().get_size();
	tensor_ctrl<double> tctrl(t);
	double *p = tctrl.req_dataptr();
	for(size_t i=0; i<sz; i++) p[i] += m_v*c;
	tctrl.ret_dataptr(p);
}

} // namespace libtensor

