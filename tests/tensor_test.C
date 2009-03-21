#include "tensor_ctrl.h"
#include "tensor_test.h"

namespace libtensor {

typedef tensor<2, double, libvmm::std_allocator<double> > tensor2_d;
typedef tensor<2, int, libvmm::std_allocator<int> > tensor2_int;

void tensor_test::perform() throw(libtest::test_exception) {
	test_ctor();
	test_immutable();
	test_operation();
}

void tensor_test::test_ctor() throw(libtest::test_exception) {
	index<2> i1, i2;
	i2[0] = 2; i2[1] = 3;
	index_range<2> ir(i1, i2);
	dimensions<2> d1(ir);
	tensor2_d t1(d1);

	if(t1.is_immutable()) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"A new tensor must be mutable (t1)");
	}

	if(t1.get_dims()[0] != 3) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 0 (t1)");
	}
	if(t1.get_dims()[1] != 4) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 1 (t1)");
	}

	tensor2_d t2(t1);

	if(t2.is_immutable()) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"A new tensor must be mutable (t2)");
	}

	if(t2.get_dims()[0] != 3) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 0 (t2)");
	}
	if(t2.get_dims()[1] != 4) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 1 (t2)");
	}

	tensor_i<2,double> *pt2 = &t2;
	tensor2_d t3(*pt2);

	if(t3.is_immutable()) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"A new tensor must be mutable (t3)");
	}

	if(t3.get_dims()[0] != 3) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 0 (t3)");
	}
	if(t3.get_dims()[1] != 4) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 1 (t3)");
	}
}

void tensor_test::test_op_chk_imm::perform(tensor_i<2,int> &t)
	throw(exception) {
	m_ok = false;
	dimensions<2> d(t.get_dims());
	int *ptr = NULL;
	tensor_ctrl<2,int> tctrl(t);
	try {
		ptr = tctrl.req_dataptr();
	} catch(exception e) {
		m_ok = true;
	}
	if(ptr) {
		tctrl.ret_dataptr(ptr);
		ptr = NULL;
	}
}

void tensor_test::test_immutable() throw(libtest::test_exception) {
	index<2> i1, i2;
	i2[0] = 2; i2[1] = 3;
	index_range<2> ir(i1, i2);
	dimensions<2> d1(ir);
	tensor2_int t1(d1);

	if(t1.is_immutable()) {
		fail_test("tensor_test::test_immutable()", __FILE__, __LINE__,
			"New tensor t1 is not mutable");
	}

	t1.set_immutable();

	if(!t1.is_immutable()) {
		fail_test("tensor_test::test_immutable()", __FILE__, __LINE__,
			"Setting t1 immutable failed");
	}

	test_op_chk_imm op;
	op.perform(t1);
	if(!op.is_ok()) {
		fail_test("tensor_test::test_immutable()", __FILE__, __LINE__,
			"Requesting non-const pointer in t1 must fail");
	}
}

void tensor_test::test_op_set_int::perform(tensor_i<2,int> &t)
	throw(exception) {
	dimensions<2> d(t.get_dims());
	tensor_ctrl<2,int> tctrl(t);
	int *ptr = tctrl.req_dataptr();
	if(ptr) {
		for(size_t i=0; i<d.get_size(); i++) ptr[i] = m_val;
	}
	tctrl.ret_dataptr(ptr);
}

void tensor_test::test_op_chkset_int::perform(tensor_i<2,int> &t)
	throw(exception) {
	m_ok = true;
	dimensions<2> d(t.get_dims());
	tensor_ctrl<2,int> tctrl(t);
	const int *ptr = tctrl.req_const_dataptr();
	if(ptr) {
		for(size_t i=0; i<d.get_size(); i++)
			m_ok = m_ok && (ptr[i]==m_val);
	}
	tctrl.ret_dataptr(ptr);
}

void tensor_test::test_op_chk_dblreq::perform(tensor_i<2,int> &t)
	throw(exception) {
	m_ok = true;
	tensor_ctrl<2,int> tctrl(t);

	// After rw-checkout, ro-checkout is not allowed
	int *ptr = tctrl.req_dataptr();
	try {
		int *ptr2 = tctrl.req_dataptr();
		m_ok = false;
	} catch(exception e) {
	}
	try {
		const int *ptr2 = tctrl.req_const_dataptr();
		m_ok = false;
	} catch(exception e) {
	}
	tctrl.ret_dataptr(ptr);

	// After ro-checkout, rw-checkout is not allowed
	const int *const_ptr = tctrl.req_const_dataptr();
	try {
		int *ptr2 = tctrl.req_dataptr();
		m_ok = false;
	} catch(exception e) {
	}

	// Multiple ro-checkouts are allowed
	try {
		const int *ptr2 = tctrl.req_const_dataptr();
		tctrl.ret_dataptr(ptr2);
	} catch(exception e) {
		m_ok = false;
	}
	tctrl.ret_dataptr(const_ptr);
}

void tensor_test::test_operation() throw(libtest::test_exception) {
	index<2> i1, i2;
	i2[0] = 2; i2[1] = 3;
	index_range<2> ir(i1, i2);
	dimensions<2> d1(ir);
	tensor2_int t1(d1);

	test_op_set_int op1(1), op100(100);
	test_op_chkset_int chkop1(1), chkop100(100);

	op1.perform(t1);
	chkop1.perform(t1);
	if(!chkop1.is_ok()) {
		fail_test("tensor_test::test_operation()", __FILE__, __LINE__,
			"Operation failed to set all elements to 1 (t1)");
	}
	op100.perform(t1);
	chkop100.perform(t1);
	if(!chkop100.is_ok()) {
		fail_test("tensor_test::test_operation()", __FILE__, __LINE__,
			"Operation failed to set all elements to 100 (t1)");
	}

	test_op_chk_dblreq op_dblreq;
	try {
		op_dblreq.perform(t1);
	} catch(exception e) {
		fail_test("tensor_test::test_operation()", __FILE__, __LINE__,
			e.what());
	}
	if(!op_dblreq.is_ok()) {
		fail_test("tensor_test::test_operation()", __FILE__, __LINE__,
			"Double requests for data must cause an exception");
	}

}

} // namespace libtensor

