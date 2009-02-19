#include "tensor_test.h"

namespace libtensor {

typedef tensor<double, libvmm::std_allocator<double> > tensor_d;
typedef tensor<int, libvmm::std_allocator<int> > tensor_int;

void tensor_test::perform() throw(libtest::test_exception) {
	test_ctor();
	test_immutable();
	test_operation();
}

void tensor_test::test_ctor() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0] = 2; i2[1] = 3;
	index_range ir(i1, i2);
	dimensions d1(ir);
	tensor_d t1(d1);

	if(t1.is_immutable()) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"A new tensor must be mutable (t1)");
	}

	if(t1.get_dims().get_order() != 2) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor order (t1)");
	}
	if(t1.get_dims()[0] != 2) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 0 (t1)");
	}
	if(t1.get_dims()[1] != 3) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 1 (t1)");
	}

	tensor_d t2(t1);

	if(t2.is_immutable()) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"A new tensor must be mutable (t2)");
	}

	if(t2.get_dims().get_order() != 2) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor order (t2)");
	}
	if(t2.get_dims()[0] != 2) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 0 (t2)");
	}
	if(t2.get_dims()[1] != 3) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 1 (t2)");
	}

	tensor_i<double> *pt2 = &t2;
	tensor_d t3(*pt2);

	if(t3.is_immutable()) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"A new tensor must be mutable (t3)");
	}

	if(t3.get_dims().get_order() != 2) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor order (t3)");
	}
	if(t3.get_dims()[0] != 2) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 0 (t3)");
	}
	if(t3.get_dims()[1] != 3) {
		fail_test("tensor_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect tensor dimension 1 (t3)");
	}
}

void tensor_test::test_immutable() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0] = 2; i2[1] = 3;
	index_range ir(i1, i2);
	dimensions d1(ir);
	tensor_d t1(d1);

	if(t1.is_immutable()) {
		fail_test("tensor_test::test_immutable()", __FILE__, __LINE__,
			"New tensor t1 is not mutable");
	}

	t1.set_immutable();

	if(!t1.is_immutable()) {
		fail_test("tensor_test::test_immutable()", __FILE__, __LINE__,
			"Setting t1 immutable failed");
	}
}

void tensor_test::test_op_set_int::perform(tensor_i<int> &t) throw(exception) {
	dimensions d(t.get_dims());
	permutation p(d.get_order());
	int *ptr = req_dataptr(t, p);
	if(ptr) {
		for(size_t i=0; i<d.get_size(); i++) ptr[i] = m_val;
	}
	ret_dataptr(t, ptr);
}

void tensor_test::test_op_chkset_int::perform(tensor_i<int> &t)
	throw(exception) {
	m_ok = true;
	dimensions d(t.get_dims());
	permutation p(d.get_order());
	const int *ptr = req_const_dataptr(t, p);
	if(ptr) {
		for(size_t i=0; i<d.get_size(); i++)
			m_ok = m_ok && (ptr[i]==m_val);
	}
	ret_dataptr(t, ptr);
}

void tensor_test::test_op_chk_dblreq::perform(tensor_i<int> &t)
	throw(exception) {
	m_ok = true;
	permutation p(t.get_dims().get_order());

	int *ptr = req_dataptr(t, p);
	try {
		int *ptr2 = req_dataptr(t, p);
		m_ok = false;
	} catch(exception e) {
	}
	try {
		const int *ptr2 = req_const_dataptr(t, p);
		m_ok = false;
	} catch(exception e) {
	}
	ret_dataptr(t, ptr);

	const int *const_ptr = req_const_dataptr(t, p);
	try {
		int *ptr2 = req_dataptr(t, p);
		m_ok = false;
	} catch(exception e) {
	}
	try {
		const int *ptr2 = req_const_dataptr(t, p);
		m_ok = false;
	} catch(exception e) {
	}
	ret_dataptr(t, const_ptr);
}

void tensor_test::test_operation() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0] = 2; i2[1] = 3;
	index_range ir(i1, i2);
	dimensions d1(ir);
	tensor_int t1(d1);

	test_op_set_int op1(1), op100(100);
	test_op_chkset_int chkop1(1), chkop100(100);

	t1.operation(op1);
	t1.operation(chkop1);
	if(!chkop1.is_ok()) {
		fail_test("tensor_test::test_operation()", __FILE__, __LINE__,
			"Operation failed to set all elements to 1 (t1)");
	}
	t1.operation(op100);
	t1.operation(chkop100);
	if(!chkop100.is_ok()) {
		fail_test("tensor_test::test_operation()", __FILE__, __LINE__,
			"Operation failed to set all elements to 100 (t1)");
	}

	test_op_chk_dblreq op_dblreq;
	try {
		t1.operation(op_dblreq);
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

