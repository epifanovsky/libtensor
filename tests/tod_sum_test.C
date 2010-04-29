#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/core/tensor_ctrl.h>
#include <libtensor/tod/tod_additive.h>
#include <libtensor/tod/tod_sum.h>
#include "tod_sum_test.h"

namespace libtensor {

typedef tensor<4, double, libvmm::std_allocator<double> > tensor4_d;

void tod_sum_test::perform() throw(libtest::test_exception) {

	test_1();
}


namespace tod_sum_test_ns {


//!	Assigns every element its number
class testop_set : public tod_additive<4> {
public:
	virtual void prefetch() { }

	virtual void perform(tensor_i<4,double> &t) {

		size_t sz = t.get_dims().get_size();
		tensor_ctrl<4, double> tctrl(t);
		double *p = tctrl.req_dataptr();
		for(size_t i = 0; i < sz; i++) p[i] = (double)i;
		tctrl.ret_dataptr(p);
	}

	virtual void perform(tensor_i<4,double> &t, double c) {

		size_t sz = t.get_dims().get_size();
		tensor_ctrl<4, double> tctrl(t);
		double *p = tctrl.req_dataptr();
		for(size_t i = 0; i < sz; i++) p[i] += c * (double)i;
		tctrl.ret_dataptr(p);
	}

};


//!	Adds a constant to every element
class testop_add : public tod_additive<4> {
private:
	double m_v;

public:
	testop_add(double v) : m_v(v) { }

	virtual void prefetch() { }

	virtual void perform(tensor_i<4, double> &t) {

		size_t sz = t.get_dims().get_size();
		tensor_ctrl<4, double> tctrl(t);
		double *p = tctrl.req_dataptr();
		for(size_t i = 0; i < sz; i++) p[i] += m_v;
		tctrl.ret_dataptr(p);
	}

	virtual void perform(tensor_i<4, double> &t, double c) {

		size_t sz = t.get_dims().get_size();
		tensor_ctrl<4, double> tctrl(t);
		double *p = tctrl.req_dataptr();
		for(size_t i = 0; i < sz; i++) p[i] += m_v * c;
		tctrl.ret_dataptr(p);
	}

};


} // namespace tod_sum_test_ns
namespace ns = tod_sum_test_ns;


void tod_sum_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "tod_sum_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 3; i2[1] = 3; i2[2] = 4; i2[3] = 4;
	dimensions<4> dims(index_range<4>(i1, i2));
	tensor<4, double, allocator_t> t(dims);

	ns::testop_set setop;
	tod_sum<4> op(setop);
	ns::testop_add add1(1.0), add2(2.0);
	op.add_op(add1, 1.0);
	op.add_op(add2, 1.0);
	op.perform(t);

	bool ok = true;
	{
		tensor_ctrl<4, double> tctrl(t);
		const double *p = tctrl.req_const_dataptr();
		size_t sz = dims.get_size();
		for(size_t i = 0; i < sz; i++) {
			if(p[i] != (double)i + 3.0) {
				ok = false;
				break;
			}
		}
		tctrl.ret_dataptr(p);
	}

	if(!ok) {
		fail_test(testname, __FILE__, __LINE__, "Set operation failed");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
