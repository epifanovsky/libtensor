#include <sstream>
#include <libtensor.h>
#include "contraction2_test.h"

namespace libtensor {


void contraction2_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
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


void contraction2_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "contraction2_test::test_4()";

	try {

	permutation<2> perma;
	permutation<4> permb;
	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);
	contraction2<1, 3, 1> contr_ref(contr);
	contr.permute_a(perma);
	contr.permute_b(permb);

	const sequence<10, size_t> &conn = contr.get_conn();
	const sequence<10, size_t> &conn_ref = contr_ref.get_conn();
	for(size_t i = 0; i < 10; i++) {
		if(conn[i] != conn_ref[i]) {
			std::ostringstream ss;
			ss << "Incorrect connection at position " << i << ": "
				<< conn[i] << " vs. " << conn_ref[i]
				<< " (ref).";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contraction2_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "contraction2_test::test_5()";

	try {

	contraction2<2, 2, 0> contr;
	sequence<8, size_t> conn_ref(0);
	conn_ref[0] = 4; conn_ref[1] = 5;
	conn_ref[2] = 6; conn_ref[3] = 7;
	conn_ref[4] = 0; conn_ref[5] = 1;
	conn_ref[6] = 2; conn_ref[7] = 3;

	const sequence<8, size_t> &conn = contr.get_conn();
	for(size_t i = 0; i < 8; i++) {
		if(conn[i] != conn_ref[i]) {
			std::ostringstream ss;
			ss << "Incorrect connection at position " << i << ": "
				<< conn[i] << " vs. " << conn_ref[i]
				<< " (ref).";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contraction2_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "contraction2_test::test_6()";

	//	c_{ab} = \sum_{ic} a_{iabc} b_{ic}

	try {

	contraction2<2, 0, 2> contr;
	contr.contract(0, 0);
	contr.contract(3, 1);

	sequence<8, size_t> conn_ref(0);
	conn_ref[0] = 3; conn_ref[1] = 4;
	conn_ref[2] = 6; conn_ref[3] = 0;
	conn_ref[4] = 1; conn_ref[5] = 7;
	conn_ref[6] = 2; conn_ref[7] = 5;

	const sequence<8, size_t> &conn = contr.get_conn();
	for(size_t i = 0; i < 8; i++) {
		if(conn[i] != conn_ref[i]) {
			std::ostringstream ss;
			ss << "Incorrect connection at position " << i << ": "
				<< conn[i] << " vs. " << conn_ref[i]
				<< " (ref).";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
