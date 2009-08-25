#include <sstream>
#include <libtensor.h>
#include "contraction2_test.h"

namespace libtensor {


void contraction2_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


void contraction2_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "contraction2_test::test_1()";

	try {

	permutation<4> perm;
	contraction2<2, 2, 2> c(perm);

	if(c.is_complete()) {
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__,
			"Empty contraction declares complete");
	}

	c.contract(2, 2);
	if(c.is_complete()) {
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__,
			"Incomplete contraction declares complete");
	}

	c.contract(3, 3);
	if(!c.is_complete()) {
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__,
			"Complete contraction declares incomplete");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contraction2_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "contraction2_test::test_2()";

	try {

	permutation<4> perm2a, perm2b;
	perm2a.permute(0, 1);
	contraction2<2, 2, 2> contr1, contr2(perm2a);
	contr1.contract(2, 2);
	contr1.contract(3, 3);
	contr1.permute_a(perm2a);
	contr1.permute_b(perm2b);
	contr2.contract(2, 2);
	contr2.contract(3, 3);

	const sequence<12, size_t> &seq1 = contr1.get_conn();
	const sequence<12, size_t> &seq2 = contr2.get_conn();

	for(size_t i = 0; i < 12; i++) {
		if(seq1[seq1[i]] != i) {
			std::ostringstream ss;
			ss << "Index connections (1) are broken at position "
				<< i << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	for(size_t i = 0; i < 12; i++) {
		if(seq2[seq2[i]] != i) {
			std::ostringstream ss;
			ss << "Index connections (2) are broken at position "
				<< i << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	bool eq = true;
	for(size_t i = 0; i < 12; i++) {
		if(seq1[i] != seq2[i]) {
			eq = false;
			break;
		}
	}
	if(!eq) {
		fail_test(testname, __FILE__, __LINE__,
			"Inconsistent contraction after permute_ab.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contraction2_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "contraction2_test::test_3()";

	try {

	permutation<4> perm2a, perm2b;
	perm2a.permute(2, 3);
	perm2b.permute(0, 1);
	contraction2<2, 2, 2> contr1, contr2;
	contr1.contract(2, 0);
	contr1.contract(3, 1);
	contr2.contract(2, 0);
	contr2.contract(3, 1);
	contr2.permute_a(perm2a);
	contr2.permute_b(perm2b);

	const sequence<12, size_t> &seq1 = contr1.get_conn();
	const sequence<12, size_t> &seq2 = contr2.get_conn();

	for(size_t i = 0; i < 12; i++) {
		if(seq1[seq1[i]] != i) {
			std::ostringstream ss;
			ss << "Index connections (1) are broken at position "
				<< i << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	for(size_t i = 0; i < 12; i++) {
		if(seq2[seq2[i]] != i) {
			std::ostringstream ss;
			ss << "Index connections (2) are broken at position "
				<< i << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	bool eq = true;
	for(size_t i = 0; i < 12; i++) {
		if(seq1[i] != seq2[i]) {
			eq = false;
			break;
		}
	}
	if(!eq) {
		fail_test(testname, __FILE__, __LINE__,
			"Inconsistent contraction after permute_ab.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
