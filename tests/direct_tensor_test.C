#include <libvmm.h>
#include "direct_tensor_test.h"

namespace libtensor {

typedef direct_tensor<int, libvmm::std_allocator<int> > tensor_int;

void direct_tensor_test::perform() throw(libtest::test_exception) {
	test_ctor();
	test_buffering();
}

void direct_tensor_test::test_op::perform(tensor_i<int> &t) throw(exception) {
}

void direct_tensor_test::test_ctor() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0]=3; i2[1]=3;
	index_range ir(i1,i2);
	dimensions d(ir);
	test_op op;
	tensor_int dt(d, op);

	tensor_i<int> &dtref(dt);
	dtref.get_dims();
}

void direct_tensor_test::test_op_set::perform(tensor_i<int> &t)
	throw(exception) {
	size_t sz = t.get_dims().get_size();
	tensor_ctrl<int> tctrl(t);
	int *p = tctrl.req_dataptr();
	for(size_t i=0; i<sz; i++) p[i] = (int)i;
	tctrl.ret_dataptr(p);
	m_count++;
}

void direct_tensor_test::test_op_chk_set::perform(tensor_i<int> &t)
	throw(exception) {
	size_t sz = t.get_dims().get_size();
	m_ok = true;
	tensor_ctrl<int> tctrl(t);
	const int *p = tctrl.req_const_dataptr();
	for(size_t i=0; i<sz; i++) if(p[i]!=(int)i) { m_ok=false; break; }
	tctrl.ret_dataptr(p);
}

void direct_tensor_test::test_buffering() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0]=3; i2[1]=3;
	index_range ir(i1,i2);
	dimensions d(ir);
	test_op_set op_set;
	test_op_chk_set op_chk_set;

	tensor_int dt(d, op_set);

	op_chk_set.perform(dt);
	if(!op_chk_set.is_ok()) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Tensor elements were set incorrectly (1)");
	}
	if(op_set.get_count()!=1) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Operation was invoked incorrectly (1)");
	}
	op_chk_set.perform(dt);
	if(!op_chk_set.is_ok()) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Tensor elements were set incorrectly (2)");
	}
	if(op_set.get_count()!=2) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Operation was invoked incorrectly (2)");
	}

	dt.enable_buffering();
	op_chk_set.perform(dt);
	if(!op_chk_set.is_ok()) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Tensor elements were set incorrectly (3)");
	}
	if(op_set.get_count()!=3) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Operation was invoked incorrectly (3)");
	}
	op_chk_set.perform(dt); // This time the set operation is not
				// invoked again
	if(!op_chk_set.is_ok()) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Tensor elements were set incorrectly (4)");
	}
	if(op_set.get_count()!=3) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Operation was invoked incorrectly (4)");
	}
	op_chk_set.perform(dt); // Set is not invoked again, it's buffered
	if(!op_chk_set.is_ok()) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Tensor elements were set incorrectly (5)");
	}
	if(op_set.get_count()!=3) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Operation was invoked incorrectly (5)");
	}

	dt.disable_buffering();
	op_chk_set.perform(dt); // Set is now called
	if(!op_chk_set.is_ok()) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Tensor elements were set incorrectly (6)");
	}
	if(op_set.get_count()!=4) {
		fail_test("direct_tensor_test::test_buffering()", __FILE__,
			__LINE__, "Operation was invoked incorrectly (6)");
	}

}

} // namespace libtensor

