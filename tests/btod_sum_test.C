#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libvmm.h>
#include <libtensor.h>
#include "btod_sum_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_sum_test::perform() throw(libtest::test_exception) {

	srand48(time(NULL));

	test_1();
	test_2();
	test_3();
}


void btod_sum_test::test_1() throw(libtest::test_exception) {

	//
	//	Single operand A + B
	//

	static const char *testname = "btod_sum_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt3_ref(bis);
	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	bt1.set_immutable();
	bt2.set_immutable();


	btod_add<2> add(bt1);
	add.add_op(bt2, 2.0);
	add.add_op(bt2);

	btod_sum<2> sum(add);
	sum.perform(bt3);
	add.perform(bt3_ref);

	compare_ref<2>::compare(testname, bt3, bt3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_sum_test::test_2() throw(libtest::test_exception) {

	//
	//	Two operands: A + B and C + D
	//

	static const char *testname = "btod_sum_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt4(bis),
		bt5(bis), bt5_ref(bis);
	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	btod_random<2>().perform(bt3);
	btod_random<2>().perform(bt4);
	btod_random<2>().perform(bt5);
	btod_copy<2>(bt5).perform(bt5_ref);
	bt1.set_immutable();
	bt2.set_immutable();
	bt3.set_immutable();
	bt4.set_immutable();

	btod_add<2> add1(bt1), add2(bt3), add_ref(bt1);
	add1.add_op(bt2);
	add2.add_op(bt4);
	add_ref.add_op(bt2);
	add_ref.add_op(bt3);
	add_ref.add_op(bt4);

	btod_sum<2> sum(add1);
	sum.add_op(add2);
	sum.perform(bt5);
	add_ref.perform(bt5_ref);

	compare_ref<2>::compare(testname, bt5, bt5_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_sum_test::test_3() throw(libtest::test_exception) {

	//
	//	Two operands: A + B and C + D
	//

	static const char *testname = "btod_sum_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt4(bis),
		bt5(bis), bt5_ref(bis);
	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	btod_random<2>().perform(bt3);
	btod_random<2>().perform(bt4);
	bt1.set_immutable();
	bt2.set_immutable();
	bt3.set_immutable();
	bt4.set_immutable();

	btod_add<2> add1(bt1), add2(bt3), add_ref(bt1);
	add1.add_op(bt2);
	add2.add_op(bt4);
	add_ref.add_op(bt2);
	add_ref.add_op(bt3, -1.0);
	add_ref.add_op(bt4, -1.0);

	btod_sum<2> sum(add1);
	sum.add_op(add2, -1.0);
	sum.perform(bt5);
	add_ref.perform(bt5_ref);

	compare_ref<2>::compare(testname, bt5, bt5_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

