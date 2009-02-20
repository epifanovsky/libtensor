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

void tensor_test::test_op_chk_imm::perform(tensor_i<int> &t) throw(exception) {
	m_ok = false;
	dimensions d(t.get_dims());
	permutation p(d.get_order());
	int *ptr = NULL;
	try {
		ptr = req_dataptr(t, p);
	} catch(exception e) {
		m_ok = true;
	}
	if(ptr) {
		ret_dataptr(t, ptr);
		ptr = NULL;
	}
}

void tensor_test::test_immutable() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0] = 2; i2[1] = 3;
	index_range ir(i1, i2);
	dimensions d1(ir);
	tensor_int t1(d1);

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

void tensor_test::test_op_chk_perm::perform(tensor_i<int> &t) throw(exception) {
	m_ok = true; 
	dimensions dim(t.get_dims());
	permutation perm(dim.get_order());

	// Assumes that the tensor has the second order

	int *p = req_dataptr(t, perm);
	int *pp = p;
	for(size_t i=0; i<dim[0]; i++) {
		for(size_t j=0; j<dim[1]; j++) {
			*pp = i+1; pp++;
		}
	}
	ret_dataptr(t, p);

	perm.permute(0, 1);
	dim.permute(perm);
	p = req_dataptr(t, perm);
	pp = p;
	for(size_t i=0; i<dim[0]; i++) {
		for(size_t j=0; j<dim[1]; j++) {
			if(*pp != j+1) m_ok = false;
			pp++;
		}
	}
	ret_dataptr(t, p);

	dim.permute(perm);
	perm.permute(0, 1);
	p = req_dataptr(t, perm);
	pp = p;
	for(size_t i=0; i<dim[0]; i++) {
		for(size_t j=0; j<dim[1]; j++) {
			if(*pp != i+1) m_ok = false;
			pp++;
		}
	}
	ret_dataptr(t, p);
}

void tensor_test::test_operation() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0] = 2; i2[1] = 3;
	index_range ir(i1, i2);
	dimensions d1(ir);
	tensor_int t1(d1);

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

	test_op_chk_perm op_perm;
	op_perm.perform(t1);
	if(!op_perm.is_ok()) {
		fail_test("tensor_test::test_operation()", __FILE__, __LINE__,
			"Requests for permuted data failed");
	}
}

} // namespace libtensor

