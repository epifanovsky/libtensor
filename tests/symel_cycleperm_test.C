#include <memory>
#include <libtensor.h>
#include "symel_cycleperm_test.h"

namespace libtensor {

void symel_cycleperm_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_equals_1();
	test_equals_2();
	test_equals_3();
}

void symel_cycleperm_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "symel_cycleperm_test::test_1()";

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> msk;
	symel_cycleperm<4, double> elem(msk, dims);
	if(!elem.get_perm().is_identity()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expecting an identity permutation.");
	}
	if(!elem.is_valid_bis(bis)) {
		fail_test(testname, __FILE__, __LINE__,
			"Applicability test failed.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void symel_cycleperm_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "symel_cycleperm_test::test_2()";

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> msk;
	msk[0] = true; msk[1] = true;
	permutation<4> perm;
	perm.permute(0, 1);
	symel_cycleperm<4, double> elem(msk, dims);
	if(!elem.get_perm().equals(perm)) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected permutation.");
	}
	if(!elem.is_valid_bis(bis)) {
		fail_test(testname, __FILE__, __LINE__,
			"Applicability test failed.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void symel_cycleperm_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "symel_cycleperm_test::test_3()";

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2).permute(2, 3);
	symel_cycleperm<4, double> elem(msk, dims);
	if(!elem.get_perm().equals(perm)) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected permutation.");
	}
	if(!elem.is_valid_bis(bis)) {
		fail_test(testname, __FILE__, __LINE__,
			"Applicability test failed.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void symel_cycleperm_test::test_equals_1() throw(libtest::test_exception) {

	static const char *testname = "symel_cycleperm_test::test_equals_1()";

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	mask<4> msk;
	symel_cycleperm<4, double> elem(msk, dims);
	if(!elem.equals(elem)) {
		fail_test(testname, __FILE__, __LINE__,
			"Self-equality test failed.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void symel_cycleperm_test::test_equals_2() throw(libtest::test_exception) {

	static const char *testname = "symel_cycleperm_test::test_equals_2()";

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	symel_cycleperm<4, double> elem1(msk, dims), elem2(msk, dims);
	if(!elem1.equals(elem2)) {
		fail_test(testname, __FILE__, __LINE__,
			"Equality test failed.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void symel_cycleperm_test::test_equals_3() throw(libtest::test_exception) {

	static const char *testname = "symel_cycleperm_test::test_equals_3()";

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	symel_cycleperm<4, double> elem1(msk, dims);
	std::auto_ptr< symmetry_element_i<4, double> > elem2(elem1.clone());
	if(!elem1.equals(*elem2)) {
		fail_test(testname, __FILE__, __LINE__,
			"Clone equality test failed.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
