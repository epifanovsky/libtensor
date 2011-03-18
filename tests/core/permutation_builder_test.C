#include <libtensor/core/permutation_builder.h>
#include "permutation_builder_test.h"

namespace libtensor {


void permutation_builder_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
}


void permutation_builder_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "permutation_builder_test::test_1()";

	try {

	char seq_ab[] = { 'a', 'b' };
	char seq_ba[] = { 'b', 'a' };

	permutation_builder<2> pb1(seq_ab, seq_ab);
	permutation<2> p1;
	if(!pb1.get_perm().equals(p1)) {
		fail_test(testname, __FILE__, __LINE__, "Test (ab, ab) failed");
	}

	permutation_builder<2> pb2(seq_ab, seq_ba);
	permutation<2> p2; p2.permute(0, 1);
	if(!pb2.get_perm().equals(p2)) {
		fail_test(testname, __FILE__, __LINE__, "Test (ab, ba) failed");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void permutation_builder_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "permutation_builder_test::test_2()";

	try {

	char seq_abcd[] = { 'a', 'b', 'c', 'd' };
	char seq_acdb[] = { 'a', 'c', 'd', 'b' };
	sequence<4, char> seq_abcd_1('\0');
	permutation_builder<4> pb3(seq_abcd, seq_acdb);
	permutation<4> p3; p3.permute(1, 2).permute(1, 3);
	for(size_t i = 0; i < 4; i++) seq_abcd_1[i] = seq_acdb[i];
	p3.apply(seq_abcd_1);
	for(size_t i = 0; i < 4; i++) {
		if(seq_abcd_1[i] != seq_abcd[i]) {
			fail_test(testname, __FILE__, __LINE__,
				"Wrong reference in (abcd, acdb).");
		}
	}
	if(!pb3.get_perm().equals(p3)) {
		fail_test(testname, __FILE__, __LINE__,
			"Test (abcd, acdb) failed");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Uses a permutation [ab->ba] and a map [ab->ba].
		The mapped permutation is [ab->ba] and the expected
		result is [01->10].
 **/
void permutation_builder_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "permutation_builder_test::test_3()";

	try {

	char seq_ab[] = { 'a', 'b' };
	char seq_ba[] = { 'b', 'a' };
	permutation<2> map; map.permute(0, 1);
	permutation_builder<2> pb3(seq_ab, seq_ba, map);
	permutation<2> p3; p3.permute(0, 1);
	if(!pb3.get_perm().equals(p3)) {
		fail_test(testname, __FILE__, __LINE__,
			"Test (ab, ba) with (a->b, b->a) failed");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Uses a permutation [abcd->acdb] and a map [abcd->bacd].
		The mapped permutation is [abcd->cbda] and the expected
		result is [0123->3102].
 **/
void permutation_builder_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "permutation_builder_test::test_4()";

	try {

	char seq_abcd[] = { 'a', 'b', 'c', 'd' };
	char seq_acdb[] = { 'a', 'c', 'd', 'b' };
	permutation<4> map; map.permute(0, 1);
	permutation_builder<4> pb3(seq_abcd, seq_acdb, map);
	permutation<4> p3; p3.permute(2, 3).permute(0, 2);
	if(!pb3.get_perm().equals(p3)) {
		fail_test(testname, __FILE__, __LINE__,
			"Test (abcd, acdb) with (a->b, b->a) failed");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Uses a permutation [abcd->bcda] and a map [abcd->bcda].
		The mapped permutation is [abcd->bcda] and the expected
		result is [0123->1230].
 **/
void permutation_builder_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "permutation_builder_test::test_5()";

	try {

	char seq_abcd[] = { 'a', 'b', 'c', 'd' };
	char seq_bcda[] = { 'b', 'c', 'd', 'a' };
	permutation<4> map; map.permute(0, 1).permute(1, 2).permute(2, 3);
	permutation_builder<4> pb3(seq_abcd, seq_bcda, map);
	permutation<4> p3; p3.permute(0, 1).permute(1, 2).permute(2, 3);
	p3.invert(); //?
	if(!pb3.get_perm().equals(p3)) {
		fail_test(testname, __FILE__, __LINE__,
			"Test (abcd, bcda) with (abcd->bcda) failed");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Uses a permutation [abcd->bcda] and a map [abcd->acdb].
		The mapped permutation is [abcd->cdba] and the expected
		result is [0123->2310].
 **/
void permutation_builder_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "permutation_builder_test::test_6()";

	try {

	char seq_abcd[] = { 'a', 'b', 'c', 'd' };
	char seq_bcda[] = { 'b', 'c', 'd', 'a' };
	permutation<4> map; map.permute(1, 2).permute(2, 3);
	permutation_builder<4> pb3(seq_abcd, seq_bcda, map);
	permutation<4> p3; p3.permute(1, 2).permute(3, 1).permute(0, 3);
	if(!pb3.get_perm().equals(p3)) {
		fail_test(testname, __FILE__, __LINE__,
			"Test (abcd, bcda) with (abcd->cdba) failed");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
