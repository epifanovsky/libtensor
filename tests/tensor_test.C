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

void tensor_test::test_op_1_int::perform(tensor_i<int> &t) throw(exception) {
}

void tensor_test::test_operation() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0] = 2; i2[1] = 3;
	index_range ir(i1, i2);
	dimensions d1(ir);
	tensor_int t1(d1);

	test_op_1_int op1;
	t1.operation(op1);
}

} // namespace libtensor

