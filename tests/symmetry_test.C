#include <sstream>
#include <libtensor.h>
#include "symmetry_test.h"

namespace libtensor {


void symmetry_test::perform() throw(libtest::test_exception) {

	test_equals_1();
	test_equals_2();
	test_equals_3();
	test_equals_4();
}


void symmetry_test::test_equals_1() throw(libtest::test_exception) {

	//
	//	Two symmetries consist of the same elements
	//	added in the same order
	//

	static const char *testname = "symmetry_test::test_equals_1()";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);

	symmetry<4, double> sym1(bis), sym2(bis);
	sym1.add_element(cycle1);
	sym1.add_element(cycle2);
	sym2.add_element(cycle1);
	sym2.add_element(cycle2);

	if(!sym1.equals(sym2)) {
		fail_test(testname, __FILE__, __LINE__, "!sym1.equals(sym2)");
	}
	if(!sym2.equals(sym1)) {
		fail_test(testname, __FILE__, __LINE__, "!sym2.equals(sym1)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symmetry_test::test_equals_2() throw(libtest::test_exception) {

	//
	//	Two symmetries consist of the same elements
	//	added in a different order
	//

	static const char *testname = "symmetry_test::test_equals_2()";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);

	symmetry<4, double> sym1(bis), sym2(bis);
	sym1.add_element(cycle1);
	sym1.add_element(cycle2);
	sym2.add_element(cycle2);
	sym2.add_element(cycle1);

	if(!sym1.equals(sym2)) {
		fail_test(testname, __FILE__, __LINE__, "!sym1.equals(sym2)");
	}
	if(!sym2.equals(sym1)) {
		fail_test(testname, __FILE__, __LINE__, "!sym2.equals(sym1)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symmetry_test::test_equals_3() throw(libtest::test_exception) {

	//
	//	Two different symmetries: S2 < S1
	//

	static const char *testname = "symmetry_test::test_equals_3()";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);

	symmetry<4, double> sym1(bis), sym2(bis);
	sym1.add_element(cycle1);
	sym1.add_element(cycle2);
	sym2.add_element(cycle1);

	if(sym1.equals(sym2)) {
		fail_test(testname, __FILE__, __LINE__, "sym1.equals(sym2)");
	}
	if(sym2.equals(sym1)) {
		fail_test(testname, __FILE__, __LINE__, "sym2.equals(sym1)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symmetry_test::test_equals_4() throw(libtest::test_exception) {

	//
	//	Two different symmetries: S1 and S2 have no common elements
	//

	static const char *testname = "symmetry_test::test_equals_4()";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);

	symmetry<4, double> sym1(bis), sym2(bis);
	sym1.add_element(cycle1);
	sym2.add_element(cycle2);

	if(sym1.equals(sym2)) {
		fail_test(testname, __FILE__, __LINE__, "sym1.equals(sym2)");
	}
	if(sym2.equals(sym1)) {
		fail_test(testname, __FILE__, __LINE__, "sym2.equals(sym1)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
