#include <sstream>
#include <libtensor.h>
#include "btod_dotprod_test.h"

namespace libtensor {


void btod_dotprod_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
}


void btod_dotprod_test::test_1() throw(libtest::test_exception) {

	//
	//	Single block, both arguments are non-zero
	//

	static const char *testname = "btod_dotprod_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

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

	typedef libvmm::std_allocator<double> allocator_t;

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

	typedef libvmm::std_allocator<double> allocator_t;

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

	typedef libvmm::std_allocator<double> allocator_t;

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


} // namespace libtensor
