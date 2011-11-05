#include <cmath>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/symmetry.h>
#include <libtensor/core/symmetry_element_set.h>
#include <libtensor/btod/btod_print.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btod/btod_read.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/symmetry_element_set_adapter.h>
#include "btod_print_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_print_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
	test_7();
	test_8();
	test_9();
	test_10();
}


void btod_print_test::test_1() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim) with one block
	//

	static const char *testname = "btod_print_test::test_1()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 4; i2[1] = 5;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);

	btod_random<2>().perform(bt);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<2>(ioss).perform(bt);

	btod_read<2>(ioss).perform(bt_ref);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_print_test::test_2() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim), two blocks along each dimension
	//

	static const char *testname = "btod_print_test::test_2()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 4; i2[1] = 5;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk1, msk2;
	msk1[0] = true; msk2[1] = true;
	bis.split(msk1, 2);
	bis.split(msk2, 3);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);

	btod_random<2>().perform(bt);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<2>(ioss).perform(bt);

	btod_read<2>(ioss).perform(bt_ref);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_print_test::test_3() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim) with one zero block
	//

	static const char *testname = "btod_print_test::test_3()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 4; i2[1] = 5;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<2>(ioss).perform(bt);
	btod_read<2>(ioss).perform(bt_ref);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_print_test::test_4() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim), two blocks along each dimension,
	//	zero off-diagonal blocks
	//

	static const char *testname = "btod_print_test::test_4()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 4; i2[1] = 5;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk1, msk2;
	msk1[0] = true; msk2[1] = true;
	bis.split(msk1, 2);
	bis.split(msk2, 3);

	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);

	index<2> ii;
	btod_random<2> rand;
	rand.perform(bt, ii);
	ii[0] = 1; ii[1] = 1;
	rand.perform(bt, ii);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<2>(ioss).perform(bt);
	btod_read<2>(ioss).perform(bt_ref);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_print_test::test_5() throw(libtest::test_exception) {

	//
	//	Block tensor (4-dim) with one block
	//

	static const char *testname = "btod_print_test::test_5()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 4; i2[1] = 5; i2[2] = 4; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	block_tensor<4, double, allocator_t> bt(bis), bt_ref(bis);
	btod_random<4>().perform(bt);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<4>(ioss).perform(bt);

	btod_read<4>(ioss).perform(bt_ref);

	compare_ref<4>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_print_test::test_6() throw(libtest::test_exception) {

	//
	//	Block tensor (4-dim), two blocks along each dimension (with symmetry)
	//

	static const char *testname = "btod_print_test::test_6()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 4; i2[1] = 5; i2[2] = 4; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m1, m2;
	m1[0] = true; m1[2] = true;
	m2[1] = true; m2[3] = true;
	bis.split(m1, 2); bis.split(m2, 3);

	block_tensor<4, double, allocator_t> bt(bis), bt_ref(bis);
	btod_random<4>().perform(bt);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<4>(ioss).perform(bt);

	btod_read<4>(ioss).perform(bt_ref);

	compare_ref<4>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_print_test::test_7() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim), two blocks along each dimension,
	//	the size of each block is 1x1
	//

	static const char *testname = "btod_print_test::test_7()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 1; i2[1] = 1;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk1, msk2;
	msk1[0] = true; msk2[1] = true;
	bis.split(msk1, 1); bis.split(msk2, 1);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);

	btod_random<2>().perform(bt);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<2>(ioss).perform(bt);

	btod_read<2>(ioss).perform(bt_ref);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_print_test::test_8() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim), two blocks along each dimension,
	//	the sizes of blocks are 1 and 2
	//

	static const char *testname = "btod_print_test::test_8()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk1, msk2;
	msk1[0] = true; msk2[1] = true;
	bis.split(msk1, 1); bis.split(msk2, 1);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);
	btod_random<2>().perform(bt);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<2>(ioss).perform(bt);

	btod_read<2>(ioss).perform(bt_ref);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void btod_print_test::test_9() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim), two blocks along each dimension,
	//	the sizes of blocks are 1 and 2, permutational symmetry
	//

	static const char *testname = "btod_print_test::test_9()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk1, msk2;
	msk1[0] = true; msk2[1] = true;
	bis.split(msk1, 1); bis.split(msk2, 1);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);

	{
		block_tensor_ctrl<2, double> ctrl(bt), ctrl_ref(bt_ref);
		se_perm<2, double> sp(permutation<2>().permute(0, 1), true);
		ctrl.req_symmetry().insert(sp);
		ctrl_ref.req_symmetry().insert(sp);
	}
	btod_random<2>().perform(bt);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<2>(ioss).perform(bt);

	btod_read<2>(ioss).perform(bt_ref);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void btod_print_test::test_10() throw(libtest::test_exception) {

	//
	//	Block tensor (4-dim), two blocks along each dimension,
	//	permutational anti-symmetry.
	//

	static const char *testname = "btod_print_test::test_10()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 4; i2[1] = 4; i2[2] = 5; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true; msk2[2] = true; msk2[3] = true;
	bis.split(msk1, 2); bis.split(msk2, 3);
	block_tensor<4, double, allocator_t> bt(bis), bt_ref(bis);
	{
		block_tensor_ctrl<4, double> ctrl(bt), ctrl_ref(bt_ref);
		se_perm<4, double> sp1(permutation<4>().permute(0, 1), false);
		se_perm<4, double> sp2(permutation<4>().permute(2, 3), false);
		ctrl.req_symmetry().insert(sp1);
		ctrl.req_symmetry().insert(sp2);
		ctrl_ref.req_symmetry().insert(sp1);
		ctrl_ref.req_symmetry().insert(sp2);
	}

	btod_random<4>().perform(bt);
	bt.set_immutable();

	std::stringstream ioss;
	btod_print<4>(ioss).perform(bt);

	btod_read<4>(ioss).perform(bt_ref);

	compare_ref<4>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
