#include "tod_copy_test.h"
#include "tensor.h"

namespace libtensor {

void tod_copy_test::perform() throw(libtest::test_exception) {
	test_exc();

	index<2> i2a, i2b; i2b[0]=10; i2b[1]=12;
	index_range<2> ir2(i2a, i2b); dimensions<2> dims2(ir2);
	test_operation(dims2);
}

typedef tensor<4, double, libvmm::std_allocator<double> > tensor4;
typedef tensor_ctrl<4,double> tensor4_ctrl;

void tod_copy_test::test_exc() throw(libtest::test_exception) {
	index<4> i1, i2, i3;
	i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
	i3[0]=3; i3[1]=3; i3[2]=3; i3[3]=3;
	index_range<4> ir1(i1,i2), ir2(i1,i3);
	dimensions<4> dim1(ir1), dim2(ir2);
	tensor4 t1(dim1), t2(dim2);

	bool ok = false;
	try {
		tod_copy<4> tc(t1); tc.perform(t2);
	} catch(exception e) {
		ok = true;
	}

	if(!ok) {
		fail_test("tod_copy_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception with heterogeneous arguments");
	}
}

} // namespace libtensor

