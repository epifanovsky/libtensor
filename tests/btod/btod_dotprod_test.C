#include <cmath>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/btod/btod_dotprod.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_dotprod.h>
#include "btod_dotprod_test.h"

namespace libtensor {


void btod_dotprod_test::perform() throw(libtest::test_exception) {

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
	test_11();
	test_12();
}


void btod_dotprod_test::test_1() throw(libtest::test_exception) {

	//
	//	Single block, both arguments are non-zero
	//

	static const char *testname = "btod_dotprod_test::test_1()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 8; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt1(bis), bt2(bis);

	//	Fill in random data

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);

	//	Compute the dot product

	double d = btod_dotprod<2>(bt1, bt2).calculate();

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	double d_ref = tod_dotprod<2>(t1, t2).calculate();

	//	Compare

	if(fabs(d - d_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result does not match reference: " << d << " vs. "
			<< d_ref << " (ref), " << d - d_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_dotprod_test::test_2() throw(libtest::test_exception) {

	//
	//	Single block, one of the arguments is zero
	//

	static const char *testname = "btod_dotprod_test::test_2()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 8; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt1(bis), bt2(bis);

	//	Fill in random data

	btod_random<2>().perform(bt1);

	//	Compute the dot product

	double d = btod_dotprod<2>(bt1, bt2).calculate();

	//	Compare

	double d_ref = 0.0;
	if(fabs(d) != 0.0) {
		std::ostringstream ss;
		ss << "Result does not match reference: " << d << " vs. "
			<< d_ref << " (ref), " << d - d_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_dotprod_test::test_3() throw(libtest::test_exception) {

	//
	//	Two blocks in each dimension, both arguments are non-zero
	//

	static const char *testname = "btod_dotprod_test::test_3()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 8; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m1, m2;
	m1[0] = true; m2[1] = true;
	bis.split(m1, 5);
	bis.split(m2, 2);
	block_tensor<2, double, allocator_t> bt1(bis), bt2(bis);

	//	Fill in random data

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);

	//	Compute the dot product

	double d = btod_dotprod<2>(bt1, bt2).calculate();

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	double d_ref = tod_dotprod<2>(t1, t2).calculate();

	//	Compare

	if(fabs(d - d_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result does not match reference: " << d << " vs. "
			<< d_ref << " (ref), " << d - d_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_dotprod_test::test_4() throw(libtest::test_exception) {

	//
	//	Two blocks in each dimension, off-diagonal blocks of one of
	//	the arguments are zero
	//

	static const char *testname = "btod_dotprod_test::test_4()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 8; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m1, m2;
	m1[0] = true; m2[1] = true;
	bis.split(m1, 5);
	bis.split(m2, 2);
	block_tensor<2, double, allocator_t> bt1(bis), bt2(bis);

	//	Fill in random data

	btod_random<2>().perform(bt1);
	i1[0] = 0; i1[1] = 0;
	i2[0] = 1; i2[1] = 1;
	btod_random<2>().perform(bt2, i1);
	btod_random<2>().perform(bt2, i2);

	//	Compute the dot product

	double d = btod_dotprod<2>(bt1, bt2).calculate();

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	double d_ref = tod_dotprod<2>(t1, t2).calculate();

	//	Compare

	if(fabs(d - d_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result does not match reference: " << d << " vs. "
			<< d_ref << " (ref), " << d - d_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_dotprod_test::test_5() throw(libtest::test_exception) {

	//
	//	Two blocks in each dimension, multiple non-zero arguments
	//

	static const char *testname = "btod_dotprod_test::test_5()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 8; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m1, m2;
	m1[0] = true; m2[1] = true;
	bis.split(m1, 5);
	bis.split(m2, 2);
	block_tensor<2, double, allocator_t> bt1(bis), bt2(bis), bt3(bis),
		bt4(bis);

	//	Fill in random data

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	btod_random<2>().perform(bt3);
	btod_random<2>().perform(bt4);
	bt1.set_immutable();
	bt2.set_immutable();
	bt3.set_immutable();
	bt4.set_immutable();

	//	Compute the dot product

	btod_dotprod<2> op(bt1, bt2);
	op.add_arg(bt3, bt4);
	std::vector<double> v(2);
	op.calculate(v);
	double d1 = v[0], d2 = v[1];

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims), t3(dims), t4(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	tod_btconv<2>(bt3).perform(t3);
	tod_btconv<2>(bt4).perform(t4);
	double d1_ref = tod_dotprod<2>(t1, t2).calculate();
	double d2_ref = tod_dotprod<2>(t3, t4).calculate();

	//	Compare

	if(fabs(d1 - d1_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result 1 does not match reference: " << d1 << " vs. "
			<< d1_ref << " (ref), " << d1 - d1_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(fabs(d2 - d2_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result 2 does not match reference: " << d2 << " vs. "
			<< d2_ref << " (ref), " << d2 - d2_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_dotprod_test::test_6() throw(libtest::test_exception) {

	//
	//	Two blocks in each dimension, multiple non-zero arguments
	//

	static const char *testname = "btod_dotprod_test::test_6()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 8; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m1, m2;
	m1[0] = true; m2[1] = true;
	bis.split(m1, 5);
	bis.split(m2, 2);
	block_tensor<2, double, allocator_t> bt1(bis), bt2(bis), bt3(bis),
		bt4(bis);

	//	Fill in random data

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	btod_random<2>().perform(bt3);
	btod_random<2>().perform(bt4);
	bt1.set_immutable();
	bt2.set_immutable();
	bt3.set_immutable();
	bt4.set_immutable();

	//	Compute the dot product

	btod_dotprod<2> op(bt1, bt2);
	op.add_arg(bt3, bt1);
	op.add_arg(bt1, bt4);
	std::vector<double> v(3);
	op.calculate(v);
	double d1 = v[0], d2 = v[1], d3 = v[2];

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims), t3(dims), t4(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	tod_btconv<2>(bt3).perform(t3);
	tod_btconv<2>(bt4).perform(t4);
	double d1_ref = tod_dotprod<2>(t1, t2).calculate();
	double d2_ref = tod_dotprod<2>(t1, t3).calculate();
	double d3_ref = tod_dotprod<2>(t1, t4).calculate();

	//	Compare

	if(fabs(d1 - d1_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result 1 does not match reference: " << d1 << " vs. "
			<< d1_ref << " (ref), " << d1 - d1_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(fabs(d2 - d2_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result 2 does not match reference: " << d2 << " vs. "
			<< d2_ref << " (ref), " << d2 - d2_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(fabs(d3 - d3_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result 3 does not match reference: " << d3 << " vs. "
			<< d3_ref << " (ref), " << d3 - d3_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_dotprod_test::test_7() throw(libtest::test_exception) {

	//
	//	Three blocks in each dimension, both arguments are non-zero,
	//	permutational symmetry in both arguments
	//

	static const char *testname = "btod_dotprod_test::test_7()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);
	bis.split(m, 7);
	block_tensor<2, double, allocator_t> bt1(bis), bt2(bis);

	//	Install permutational symmetry

	{
		block_tensor_ctrl<2, double> ctrl1(bt1), ctrl2(bt2);
		se_perm<2, double> elem(permutation<2>().permute(0, 1), true);
		ctrl1.req_symmetry().insert(elem);
		ctrl2.req_symmetry().insert(elem);
	}

	//	Fill in random data

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	bt1.set_immutable();
	bt2.set_immutable();

	//	Compute the dot product

	double d = btod_dotprod<2>(bt1, bt2).calculate();

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	double d_ref = tod_dotprod<2>(t1, t2).calculate();

	//	Compare

	if(fabs(d - d_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result does not match reference: " << d << " vs. "
			<< d_ref << " (ref), " << d - d_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_dotprod_test::test_8() throw(libtest::test_exception) {

	//
	//	Three blocks in each dimension, multiple non-zero arguments,
	//	various kinds of permutational symmetry
	//

	static const char *testname = "btod_dotprod_test::test_8()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 2);
	bis.split(m, 5);
	bis.split(m, 7);

	block_tensor<4, double, allocator_t> bt1(bis), bt2(bis), bt3(bis),
		bt4(bis), bt5(bis), bt6(bis);

	//	Set up symmetry

	{
		block_tensor_ctrl<4, double> ctrl1(bt1), ctrl2(bt2), ctrl3(bt3),
			ctrl4(bt4), ctrl5(bt5), ctrl6(bt6);

		se_perm<4, double> elem1(permutation<4>().permute(0, 1).
			permute(2, 3), true);
		se_perm<4, double> elem2(permutation<4>().permute(0, 1).
			permute(2, 3), false);
		se_perm<4, double> elem3(permutation<4>().permute(0, 1), true);
		se_perm<4, double> elem4(permutation<4>().permute(0, 1).
			permute(1, 2).permute(2, 3), true);

		ctrl1.req_symmetry().insert(elem1);
		ctrl2.req_symmetry().insert(elem3);
		ctrl2.req_symmetry().insert(elem4);
		ctrl3.req_symmetry().insert(elem1);
		ctrl4.req_symmetry().insert(elem2);
		ctrl5.req_symmetry().insert(elem1);
		ctrl6.req_symmetry().insert(elem3);
	}

	//	Fill in random data

	btod_random<4>().perform(bt1);
	btod_random<4>().perform(bt2);
	btod_random<4>().perform(bt3);
	btod_random<4>().perform(bt4);
	btod_random<4>().perform(bt5);
	btod_random<4>().perform(bt6);
	bt1.set_immutable();
	bt2.set_immutable();
	bt3.set_immutable();
	bt4.set_immutable();
	bt5.set_immutable();
	bt6.set_immutable();

	//	Compute the dot product

	btod_dotprod<4> op(bt1, permutation<4>(), bt2,
		permutation<4>().permute(1, 2));
	op.add_arg(bt3, bt4);
	op.add_arg(bt5, permutation<4>().permute(0, 2).permute(1, 3), bt6,
		permutation<4>());
	std::vector<double> v(3);
	op.calculate(v);
	double d1 = v[0], d2 = v[1], d3 = v[2];

	//	Compute the reference

	tensor<4, double, allocator_t> t1(dims), t2(dims), t3(dims), t4(dims),
		t5(dims), t6(dims);
	tod_btconv<4>(bt1).perform(t1);
	tod_btconv<4>(bt2).perform(t2);
	tod_btconv<4>(bt3).perform(t3);
	tod_btconv<4>(bt4).perform(t4);
	tod_btconv<4>(bt5).perform(t5);
	tod_btconv<4>(bt6).perform(t6);
	double d1_ref = tod_dotprod<4>(t1, t2).calculate();
	double d2_ref = tod_dotprod<4>(t3, t4).calculate();
	double d3_ref = tod_dotprod<4>(t5, permutation<4>().permute(0, 2).
		permute(1, 3), t6, permutation<4>()).calculate();

	//	Compare

	if(fabs(d1 - d1_ref) > fabs(d1_ref) * 1e-14) {
		std::ostringstream ss;
		ss << "Result 1 does not match reference: " << d1 << " vs. "
			<< d1_ref << " (ref), " << d1 - d1_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(fabs(d2 - d2_ref) > 1e-11) {
		std::ostringstream ss;
		ss << "Result 2 does not match reference: " << d2 << " vs. "
			<< d2_ref << " (ref), " << d2 - d2_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(fabs(d3 - d3_ref) > fabs(d3_ref) * 1e-14) {
		std::ostringstream ss;
		ss << "Result 3 does not match reference: " << d3 << " vs. "
			<< d3_ref << " (ref), " << d3 - d3_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void btod_dotprod_test::test_9() throw(libtest::test_exception) {

	//
	//	Four blocks in each dimension, multiple non-zero arguments,
	//	permutational anti-symmetry
	//

	static const char *testname = "btod_dotprod_test::test_9()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);
	bis.split(m, 7);
	block_tensor<2, double, allocator_t> bt1(bis), bt2(bis);

	//	Set up symmetry

	{
		block_tensor_ctrl<2, double> ctrl1(bt1), ctrl2(bt2);

		se_perm<2, double> elem1(permutation<2>().permute(0, 1), false);

		ctrl1.req_symmetry().insert(elem1);
		ctrl2.req_symmetry().insert(elem1);
	}

	//	Fill in random data

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	bt1.set_immutable();
	bt2.set_immutable();

	//	Compute the dot product

	double d = btod_dotprod<2>(bt1, bt2).calculate();

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	double d_ref = tod_dotprod<2>(t1, t2).calculate();

	//	Compare

	if(fabs(d - d_ref) > fabs(d_ref) * 1e-14) {
		std::ostringstream ss;
		ss << "Result does not match reference: " << d << " vs. "
			<< d_ref << " (ref), " << d - d_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void btod_dotprod_test::test_10() throw(libtest::test_exception) {

	//
	//	Four blocks in each dimension, multiple non-zero arguments,
	//	label symmetry, but only in one block tensor
	//

	std::ostringstream tnss;
	tnss << "btod_dotprod_test::test_10()";

	typedef std_allocator<double> allocator_t;

	//
	// Setup product table
	//
	{
		point_group_table pg(tnss.str(), 2);
		pg.add_product(0, 0, 0);
		pg.add_product(0, 1, 1);
		pg.add_product(1, 1, 0);

		product_table_container::get_instance().add(pg);
	}

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);
	bis.split(m, 7);
	block_tensor<2, double, allocator_t> bt1(bis), bt2(bis);

	//	Set up symmetry

	{
		block_tensor_ctrl<2, double> ctrl1(bt1);

		se_label<2, double> l(bis.get_block_index_dims(), tnss.str());
		l.assign(m, 0, 0);
		l.assign(m, 1, 1);
		l.assign(m, 2, 0);
		l.assign(m, 3, 1);
		l.add_target(0);

		ctrl1.req_symmetry().insert(l);
	}

	//	Fill in random data

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	bt1.set_immutable();
	bt2.set_immutable();

	//	Compute the dot product

	double d = btod_dotprod<2>(bt1, bt2).calculate();

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	double d_ref = tod_dotprod<2>(t1, t2).calculate();

	//	Compare

	if(fabs(d - d_ref) > fabs(d_ref) * 1e-14) {
		std::ostringstream ss;
		ss << "Result does not match reference: " << d << " vs. "
			<< d_ref << " (ref), " << d - d_ref << " (diff).";
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		product_table_container::get_instance().erase(tnss.str());
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}

	product_table_container::get_instance().erase(tnss.str());
}


void btod_dotprod_test::test_11() throw(libtest::test_exception) {

	//
	//	Two blocks in each dimension, subtle splitting differences
	//	producing same, but not identical block index spaces
	//

	static const char *testname = "btod_dotprod_test::test_11()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis1(dims), bis2(dims);
	mask<2> m01, m10, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bis1.split(m01, 5);
	bis1.split(m10, 5);
	bis2.split(m11, 5);
	block_tensor<2, double, allocator_t> bt1(bis1), bt2(bis2);

	//	Fill in random data

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);

	//	Compute the dot product

	double d = btod_dotprod<2>(bt1, bt2).calculate();

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	double d_ref = tod_dotprod<2>(t1, t2).calculate();

	//	Compare

	if(fabs(d - d_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result does not match reference: " << d << " vs. "
			<< d_ref << " (ref), " << d - d_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_dotprod_test::test_12() throw(libtest::test_exception) {

	//
	//	Two blocks in each dimension, subtle splitting differences
	//	producing same, but not identical block index spaces
	//

	static const char *testname = "btod_dotprod_test::test_12()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis1(dims), bis2(dims);
	mask<2> m01, m10, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bis1.split(m01, 5);
	bis1.split(m10, 5);
	bis2.split(m11, 5);
	block_tensor<2, double, allocator_t> bt1(bis1), bt2(bis2);

	//	Fill in random data

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);

	//	Compute the dot product

	permutation<2> p01, p10;
	p10.permute(0, 1);
	double d = btod_dotprod<2>(bt1, p10, bt2, p01).calculate();

	//	Compute the reference

	tensor<2, double, allocator_t> t1(dims), t2(dims);
	tod_btconv<2>(bt1).perform(t1);
	tod_btconv<2>(bt2).perform(t2);
	double d_ref = tod_dotprod<2>(t1, p10, t2, p01).calculate();

	//	Compare

	if(fabs(d - d_ref) > 1e-13) {
		std::ostringstream ss;
		ss << "Result does not match reference: " << d << " vs. "
			<< d_ref << " (ref), " << d - d_ref << " (diff).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
