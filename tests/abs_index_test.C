#include <libtensor/core/abs_index.h>
#include "abs_index_test.h"

namespace libtensor {


void abs_index_test::perform() throw(libtest::test_exception) {

	test_inc_1();
	test_inc_2();
	test_last_1();
}


void abs_index_test::test_inc_1() throw(libtest::test_exception) {

	static const char *testname = "abs_index_test::test_inc_1()";

	try {

	index<4> i1, i2;
	i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
	dimensions<4> dims(index_range<4>(i1, i2));
	abs_index<4> ii(dims);

	if(!ii.inc()) {
		fail_test(testname, __FILE__, __LINE__,
			"inc(0,0,0,0) doesn't return true.");
	}
	index<4> i(ii.get_index());
	if(!(i[0]==0 && i[1]==0 && i[2]==0 && i[3]==1)) {
		fail_test(testname, __FILE__, __LINE__,
			"inc(0,0,0,0) doesn't return (0,0,0,1).");
	}
	if(ii.get_abs_index() != 1) {
		fail_test(testname, __FILE__, __LINE__,
			"inc(0,0,0,0) doesn't return 1.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void abs_index_test::test_inc_2() throw(libtest::test_exception) {

	static const char *testname = "abs_index_test::test_inc_2()";

	try {

	index<4> i1, i2;
	i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
	dimensions<4> dims(index_range<4>(i1, i2));
	i1[0] = 1; i1[1] = 1; i1[2] = 0; i1[3] = 0;
	abs_index<4> ii(i1, dims);

	if(!ii.inc()) {
		fail_test(testname, __FILE__, __LINE__,
			"inc(1,1,0,0) doesn't return true.");
	}
	index<4> i(ii.get_index());
	if(!(i[0]==1 && i[1]==1 && i[2]==0 && i[3]==1)) {
		fail_test(testname, __FILE__, __LINE__,
			"inc(0,0,0,0) doesn't return (1,1,0,1).");
	}
	if(ii.get_abs_index() != 13) {
		fail_test(testname, __FILE__, __LINE__,
			"inc(0,0,0,0) doesn't return 13.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void abs_index_test::test_last_1() throw(libtest::test_exception) {

	static const char *testname = "abs_index_test::test_last_1()";

	try {

	index<4> i1, i2;
	i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
	dimensions<4> dims(index_range<4>(i1, i2));

	i1[0] = 1; i1[1] = 1; i1[2] = 0; i1[3] = 0;
	abs_index<4> ii1(i1, dims);

	if(ii1.is_last()) {
		fail_test(testname, __FILE__, __LINE__,
			"[1,1,0,0] returns is_last() = true in [2,2,2,2]");
	}

	i1[0] = 1; i1[1] = 1; i1[2] = 1; i1[3] = 1;
	abs_index<4> ii2(i1, dims);

	if(!ii2.is_last()) {
		fail_test(testname, __FILE__, __LINE__,
			"[1,1,1,1] returns is_last() = false in [2,2,2,2]");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
