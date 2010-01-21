#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/core/tensor_ctrl.h>
#include <libtensor/tod/tod_sum.h>
#include "tod_sum_test.h"

namespace libtensor {

typedef tensor<4, double, libvmm::std_allocator<double> > tensor4_d;

void tod_sum_test::perform() throw(libtest::test_exception) {
	index<4> i1, i2;
	i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir);
	tensor4_d t(dim);

	testop_set setop;
	tod_sum<4> op(setop);
	testop_add add1(1.0), add2(2.0);
	op.add_op(add1, 1.0);
	op.add_op(add2, 1.0);
	op.perform(t);

	tensor_ctrl<4,double> tctrl(t);
	const double *p = tctrl.req_const_dataptr();
	bool ok = true;
	size_t sz = dim.get_size();
	for(size_t i=0; i<sz; i++) if(p[i]!=((double)i)+3.0) {
		ok = false; break;
	}
	tctrl.ret_dataptr(p);

	if(!ok) {
		fail_test("tod_sum_test::perform()", __FILE__, __LINE__,
			"Set operation failed");
	}
}

void tod_sum_test::testop_set::perform(tensor_i<4,double> &t) throw(exception) {
	size_t sz = t.get_dims().get_size();
	tensor_ctrl<4,double> tctrl(t);
	double *p = tctrl.req_dataptr();
	for(size_t i=0; i<sz; i++) p[i] = (double)i;
	tctrl.ret_dataptr(p);
}

void tod_sum_test::testop_set::perform(tensor_i<4,double> &t, double c)
	throw(exception) {
}

void tod_sum_test::testop_add::perform(tensor_i<4,double> &t) throw(exception) {
	size_t sz = t.get_dims().get_size();
	tensor_ctrl<4,double> tctrl(t);
	double *p = tctrl.req_dataptr();
	for(size_t i=0; i<sz; i++) p[i] += m_v;
	tctrl.ret_dataptr(p);
}

void tod_sum_test::testop_add::perform(tensor_i<4,double> &t, double c)
	throw(exception) {
	size_t sz = t.get_dims().get_size();
	tensor_ctrl<4,double> tctrl(t);
	double *p = tctrl.req_dataptr();
	for(size_t i=0; i<sz; i++) p[i] += m_v*c;
	tctrl.ret_dataptr(p);
}

} // namespace libtensor

