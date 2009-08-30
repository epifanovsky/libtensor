#include <sstream>
#include <libtensor.h>
#include "bispace_expr_test.h"

namespace libtensor {


void bispace_expr_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
	test_7();
	test_8();

	test_exc_1();
}


void bispace_expr_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "bispace_expr_test::test_1()";

	try {

	bispace<1> a(10), b(10);
	mask<2> msk, msk_ref;
	msk_ref[0] = true; msk_ref[1] = true;
	(a&b).mark_sym(0, msk);
	if(!msk.equals(msk_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask: " << msk << " vs. " << msk_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void bispace_expr_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "bispace_expr_test::test_2()";

	try {

	bispace<1> a(10), b(10), c(10);
	mask<3> msk, msk_ref;
	msk_ref[0] = true; msk_ref[1] = true; msk_ref[2] = true;
	(a&b&c).mark_sym(1, msk);
	if(!msk.equals(msk_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask: " << msk << " vs. " << msk_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void bispace_expr_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "bispace_expr_test::test_3()";

	try {

	bispace<1> a(10), b(10);
	mask<2> msk1, msk1_ref;
	mask<2> msk2, msk2_ref;
	msk1_ref[0] = true;
	msk2_ref[1] = true;
	(a|b).mark_sym(0, msk1);
	(a|b).mark_sym(1, msk2);
	if(!msk1.equals(msk1_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk2.equals(msk2_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void bispace_expr_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "bispace_expr_test::test_4()";

	try {

	bispace<1> a(10), b(10), c(10), d(10);
	mask<4> msk1, msk1_ref;
	mask<4> msk2, msk2_ref;
	mask<4> msk3, msk3_ref;
	mask<4> msk4, msk4_ref;
	msk1_ref[0] = true;
	msk2_ref[1] = true;
	msk3_ref[2] = true;
	msk4_ref[3] = true;
	((a|b)|(c|d)).mark_sym(0, msk1);
	((a|b)|(c|d)).mark_sym(1, msk2);
	((a|b)|(c|d)).mark_sym(2, msk3);
	((a|b)|(c|d)).mark_sym(3, msk4);
	if(!msk1.equals(msk1_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk2.equals(msk2_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk3.equals(msk3_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk4.equals(msk4_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void bispace_expr_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "bispace_expr_test::test_5()";

	try {

	bispace<1> a(10), b(10), c(10), d(10);
	mask<4> msk1, msk1_ref;
	mask<4> msk2, msk2_ref;
	mask<4> msk3, msk3_ref;
	mask<4> msk4, msk4_ref;
	msk1_ref[0] = true; msk1_ref[1] = true;
	msk2_ref[0] = true; msk2_ref[1] = true;
	msk3_ref[2] = true; msk3_ref[3] = true;
	msk4_ref[2] = true; msk4_ref[3] = true;
	((a&b)|(c&d)).mark_sym(0, msk1);
	((a&b)|(c&d)).mark_sym(1, msk2);
	((a&b)|(c&d)).mark_sym(2, msk3);
	((a&b)|(c&d)).mark_sym(3, msk4);
	if(!msk1.equals(msk1_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk2.equals(msk2_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk3.equals(msk3_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk4.equals(msk4_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void bispace_expr_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "bispace_expr_test::test_6()";

	try {

	bispace<1> a(10), b(10), c(10), d(10);
	mask<4> msk1, msk1_ref;
	mask<4> msk2, msk2_ref;
	mask<4> msk3, msk3_ref;
	mask<4> msk4, msk4_ref;
	msk1_ref[0] = true; msk1_ref[2] = true;
	msk2_ref[1] = true; msk2_ref[3] = true;
	msk3_ref[0] = true; msk3_ref[2] = true;
	msk4_ref[1] = true; msk4_ref[3] = true;
	((a|b)&(c|d)).mark_sym(0, msk1);
	((a|b)&(c|d)).mark_sym(1, msk2);
	((a|b)&(c|d)).mark_sym(2, msk3);
	((a|b)&(c|d)).mark_sym(3, msk4);
	if(!msk1.equals(msk1_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk2.equals(msk2_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk3.equals(msk3_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk4.equals(msk4_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void bispace_expr_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "bispace_expr_test::test_7()";

	try {

	bispace<1> a(10), b(10), c(10), d(10), e(10), f(10);
	mask<6> msk1, msk1_ref;
	mask<6> msk2, msk2_ref;
	mask<6> msk3, msk3_ref;
	mask<6> msk4, msk4_ref;
	mask<6> msk5, msk5_ref;
	mask<6> msk6, msk6_ref;
	msk1_ref[0] = true; msk1_ref[2] = true; msk1_ref[4] = true;
	msk2_ref[1] = true; msk2_ref[3] = true; msk2_ref[5] = true;
	msk3_ref[0] = true; msk3_ref[2] = true; msk3_ref[4] = true;
	msk4_ref[1] = true; msk4_ref[3] = true; msk4_ref[5] = true;
	msk5_ref[0] = true; msk5_ref[2] = true; msk5_ref[4] = true;
	msk6_ref[1] = true; msk6_ref[3] = true; msk6_ref[5] = true;
	((a|b)&(c|d)&(e|f)).mark_sym(0, msk1);
	((a|b)&(c|d)&(e|f)).mark_sym(1, msk2);
	((a|b)&(c|d)&(e|f)).mark_sym(2, msk3);
	((a|b)&(c|d)&(e|f)).mark_sym(3, msk4);
	((a|b)&(c|d)&(e|f)).mark_sym(4, msk5);
	((a|b)&(c|d)&(e|f)).mark_sym(5, msk6);
	if(!msk1.equals(msk1_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk2.equals(msk2_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk3.equals(msk3_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk4.equals(msk4_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk5.equals(msk5_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 5: " << msk5 << " vs. " << msk5_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk6.equals(msk6_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 6: " << msk6 << " vs. " << msk6_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void bispace_expr_test::test_8() throw(libtest::test_exception) {

	static const char *testname = "bispace_expr_test::test_8()";

	try {

	bispace<1> a(10), b(10), c(10), d(10), e(10), f(10);
	mask<6> msk1, msk1_ref;
	mask<6> msk2, msk2_ref;
	mask<6> msk3, msk3_ref;
	mask<6> msk4, msk4_ref;
	mask<6> msk5, msk5_ref;
	mask<6> msk6, msk6_ref;
	msk1_ref[0] = true; msk1_ref[1] = true; msk1_ref[2] = true;
	msk2_ref[0] = true; msk2_ref[1] = true; msk2_ref[2] = true;
	msk3_ref[0] = true; msk3_ref[1] = true; msk3_ref[2] = true;
	msk4_ref[3] = true; msk4_ref[4] = true; msk4_ref[5] = true;
	msk5_ref[3] = true; msk5_ref[4] = true; msk5_ref[5] = true;
	msk6_ref[3] = true; msk6_ref[4] = true; msk6_ref[5] = true;
	((a&b&c)|(d&e&f)).mark_sym(0, msk1);
	((a&b&c)|(d&e&f)).mark_sym(1, msk2);
	((a&b&c)|(d&e&f)).mark_sym(2, msk3);
	((a&b&c)|(d&e&f)).mark_sym(3, msk4);
	((a&b&c)|(d&e&f)).mark_sym(4, msk5);
	((a&b&c)|(d&e&f)).mark_sym(5, msk6);
	if(!msk1.equals(msk1_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk2.equals(msk2_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk3.equals(msk3_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk4.equals(msk4_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk5.equals(msk5_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 5: " << msk5 << " vs. " << msk5_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
	if(!msk6.equals(msk6_ref)) {
		std::ostringstream ss;
		ss << "Unexpected mask 6: " << msk6 << " vs. " << msk6_ref
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void bispace_expr_test::test_exc_1() throw(libtest::test_exception) {

	static const char *testname = "bispace_expr_test::test_exc_1()";

	try {

	bispace<1> a(10), b(20);

	bool ok = false;
	try {
		(a&b);
	} catch(expr_exception &e) {
		ok = true;
	}
	if(!ok) {
		fail_test(testname, __FILE__, __LINE__,
			"Exception expected with incompatible bispaces.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
