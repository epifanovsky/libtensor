#include <libtensor.h>
#include "mask_test.h"

namespace libtensor {


void mask_test::perform() throw(libtest::test_exception) {

	mask<2> msk1;
	mask<2> msk2(msk1);

	test_op_1();
	test_op_2();
	test_op_3();
	test_op_4();
}


/**	\test Tests the unary operator OR
 **/
void mask_test::test_op_1() throw(libtest::test_exception) {

	static const char *testname = "mask_test::test_op_1()";

	try {

	mask<4> m0, m1, m2;

	m1 |= m2;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (0000)");
	}

	m2[0] = true; m0[0] = true;
	m1 |= m2;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (1000)");
	}

	m2[2] = true; m0[2] = true;
	m2[3] = true; m0[3] = true;
	m1 |= m2;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (1011)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the binary operator OR
 **/
void mask_test::test_op_2() throw(libtest::test_exception) {

	static const char *testname = "mask_test::test_op_2()";

	try {

	mask<4> m0, m1, m2, m3;

	m1 = m2 | m3;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (0000)");
	}

	m2[0] = true; m0[0] = true;
	m1 = m2 | m3;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (1000)");
	}

	m2[0] = false; m2[1] = true; m2[3] = true; m3[3] = true;
	m0[0] = false; m0[1] = true; m0[3] = true;
	m1 = m2 | m3;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (0101)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the unary operator AND
 **/
void mask_test::test_op_3() throw(libtest::test_exception) {

	static const char *testname = "mask_test::test_op_3()";

	try {

	mask<4> m0, m1, m2;

	m1 &= m2;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (0000)");
	}

	m1[0] = true; m2[0] = true; m2[1] = true; m0[0] = true;
	m1 &= m2;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (1000)");
	}

	m1[2] = true;
	m2[2] = true; m2[3] = true;
	m0[2] = true;
	m1 &= m2;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (1010)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the binary operator AND
 **/
void mask_test::test_op_4() throw(libtest::test_exception) {

	static const char *testname = "mask_test::test_op_4()";

	try {

	mask<4> m0, m1, m2, m3;

	m1 = m2 & m3;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (0000)");
	}

	m2[0] = true; m2[1] = true;
	m3[0] = true; m3[3] = true;
	m0[0] = true;
	m1 = m2 & m3;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (1000)");
	}

	m2[0] = false; m2[1] = true; m2[3] = true;
	m3[0] = false; m3[1] = true; m3[3] = true;
	m0[0] = false; m0[1] = true; m0[3] = true;
	m1 = m2 & m3;
	if(!m1.equals(m0)) {
		fail_test(testname, __FILE__, __LINE__,
			"!m1.equals(m0) (0101)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

