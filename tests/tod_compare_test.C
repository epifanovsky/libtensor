#include <cmath>
#include <cstdlib>
#include <ctime>
#include "tod_compare_test.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

typedef tensor<4, double, libvmm::std_allocator<double> > tensor4;
typedef tensor_ctrl<4,double> tensor4_ctrl;

void tod_compare_test::perform() throw(libtest::test_exception) {
	srand48(time(NULL));

	test_exc();

	index<4> i1, i2; i2[0]=2; i2[1]=3; i2[2]=4; i2[3]=5;
	index<4> idiff; idiff[0]=0; idiff[1]=1; idiff[2]=2; idiff[3]=3;
	index_range<4> ir(i1,i2);
	dimensions<4> dim(ir);
	test_operation(dim, idiff);
	
}

void tod_compare_test::test_exc() throw(libtest::test_exception) {
	index<4> i1, i2, i3;
	i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
	i3[0]=3; i3[1]=3; i3[2]=3; i3[3]=3;
	index_range<4> ir1(i1,i2), ir2(i1,i3);
	dimensions<4> dim1(ir1), dim2(ir2);
	tensor4 t1(dim1), t2(dim2);

	bool ok = false;
	try {
		tod_compare<4> tc(t1, t2, 0);
	} catch(exception e) {
		ok = true;
	}

	if(!ok) {
		fail_test("tod_compare_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception with heterogeneous arguments");
	}
}

void tod_compare_test::test_operation(const dimensions<4> &dim,
	const index<4> &idx) throw(libtest::test_exception) {

	tensor4 t1(dim), t2(dim);
	tensor4_ctrl tctrl1(t1), tctrl2(t2);

	double *p1 = tctrl1.req_dataptr();
	double *p2 = tctrl2.req_dataptr();

	size_t sz = dim.get_size();
	for(size_t i=0; i<sz; i++) {
		p2[i] = p1[i] = drand48();
	}
	size_t diffptr = dim.abs_index(idx);
	p2[diffptr] += 1e-6;
	double diff1 = p1[diffptr];
	double diff2 = p2[diffptr];

	tctrl1.ret_dataptr(p1);
	tctrl2.ret_dataptr(p2);

	tod_compare<4> op1(t1, t2, 1e-7);
	if(op1.compare()) {
		fail_test("tod_compare_test::test_operation()", __FILE__,
			__LINE__, "tod_compare failed to find the difference");
	}
	if(dim.abs_index(op1.get_diff_index()) != diffptr) {
		fail_test("tod_compare_test::test_operation()", __FILE__,
			__LINE__, "tod_compare returned an incorrect index");
	}
	if(op1.get_diff_elem_1() != diff1 || op1.get_diff_elem_2() != diff2) {
		fail_test("tod_compare_test::test_operation()", __FILE__,
			__LINE__, "tod_compare returned an incorrect "
			"element value");
	}

	tod_compare<4> op2(t1, t2, 1e-5);
	if(!op2.compare()) {
		fail_test("tod_compare_test::test_operation()", __FILE__,
			__LINE__, "tod_compare found a difference below "
			"the threshold");
	}

}

} // namespace libtensor

