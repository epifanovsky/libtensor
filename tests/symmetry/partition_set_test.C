#include <sstream>
#include <set>
#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/partition_set.h>
#include "../compare_ref.h"
#include "partition_set_test.h"

namespace libtensor {


void partition_set_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2(true);
	test_2(false);
	test_3(true);
	test_3(false);
	test_4(true);
	test_4(false);
	test_5();

	test_add_1(true);
	test_add_1(false);
	test_add_2(true);
	test_add_2(false);
	test_add_3a(true);
	test_add_3a(false);
	test_add_3b(true);
	test_add_3b(false);
	test_add_4(true);
	test_add_4(false);
	test_add_5();
	test_add_6(true);
	test_add_6(false);
	test_add_7(true);
	test_add_7(false);
	test_add_8(true);
	test_add_8(false);

	test_permute_1();
	test_permute_2(true);
	test_permute_2(false);
	test_permute_3(true);
	test_permute_3(false);
	test_permute_4(true);
	test_permute_4(false);

	test_intersect_1();
	test_intersect_2(true);
	test_intersect_2(false);
	test_intersect_3(true);
	test_intersect_3(false);
	test_intersect_4(true);
	test_intersect_4(false);
	test_intersect_5a(true);
	test_intersect_5a(false);
	test_intersect_5b(true);
	test_intersect_5b(false);
	test_intersect_6a(true);
	test_intersect_6a(false);
	test_intersect_6b(true);
	test_intersect_6b(false);
	test_intersect_7a(true);
	test_intersect_7a(false);
	test_intersect_7b(true);
	test_intersect_7b(false);
	test_intersect_7c(true);
	test_intersect_7c(false);
	test_intersect_8(true);
	test_intersect_8(false);

	test_stabilize_1();
	test_stabilize_2(true);
	test_stabilize_2(false);
	test_stabilize_3(true);
	test_stabilize_3(false);
	test_stabilize_4(true);
	test_stabilize_4(false);

	test_merge_1();
	test_merge_2(true);
	test_merge_2(false);
	test_merge_3(true);
	test_merge_3(false);
	test_merge_4(true);
	test_merge_4(false);
}

/**	\test Create an empty partition set
 **/
void partition_set_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_1()";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);

	partition_set<2, double> pset(bis);
	pset.convert(set);

	symmetry_element_set<2, double>::iterator it = set.begin();
	if (it != set.end()) {
		fail_test(testname, __FILE__, __LINE__, "Set not empty.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Partition set created from one se_part
 **/
void partition_set_test::test_2(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_2(bool)";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	se_part<2, double> sp1(bis, m11, 2);
	sp1.add_map(i00, i11, sign);
	sp1.add_map(i01, i10, sign);
	se_part<2, double> sp2(sp1);

	symmetry_element_set<2, double> set1(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se_part<2, double>::k_sym_type);
	set1.insert(sp1);
	set2_ref.insert(sp2);

	partition_set<2, double> pset(set1);
	pset.convert(set2);

	compare_ref<2>::compare(testname, bis, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Partition set created from from multiple se_part
		(same # partitions, same sign)
 **/
void partition_set_test::test_3(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_3(bool)";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11, m01, m10;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	se_part<2, double> sp1(bis, m10, 2), sp2(bis, m01, 2);
	sp1.add_map(i00, i10, sign);
	sp2.add_map(i00, i01, sign);
	se_part<2, double> sp_ref(bis, m11, 2);
	sp_ref.add_map(i00, i10, sign);
	sp_ref.add_map(i00, i01, sign);
	sp_ref.add_map(i00, i11, true);

	symmetry_element_set<2, double> set1(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se_part<2, double>::k_sym_type);
	set1.insert(sp1);
	set1.insert(sp2);
	set2_ref.insert(sp_ref);

	partition_set<2, double> pset(set1);
	pset.convert(set2);

	compare_ref<2>::compare(testname, bis, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Partition set created from from multiple se_part
		(same # partitions, opposite sign)
 **/
void partition_set_test::test_4(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_4(bool)";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11, m01, m10;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	se_part<2, double> sp1(bis, m10, 2), sp2(bis, m01, 2);
	sp1.add_map(i00, i10, sign);
	sp2.add_map(i00, i01, ! sign);
	se_part<2, double> sp_ref(bis, m11, 2);
	sp_ref.add_map(i00, i10, sign);
	sp_ref.add_map(i00, i01, ! sign);
	sp_ref.add_map(i00, i11, false);

	symmetry_element_set<2, double> set1(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se_part<2, double>::k_sym_type);
	set1.insert(sp1);
	set1.insert(sp2);
	set2_ref.insert(sp_ref);

	partition_set<2, double> pset(set1);
	pset.convert(set2);

	compare_ref<2>::compare(testname, bis, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Partition set created from from multiple se_part
		(different # partitions)
 **/
void partition_set_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_5()";

	try {

	index<2> i1, i2;
	i2[0] = 19; i2[1] = 19;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11, m01, m10;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 5);
	bis.split(m11, 10);
	bis.split(m11, 15);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	index<2> i22, i23, i32, i33;
	i22[0] = 2; i22[1] = 2;
	i33[0] = 3; i33[1] = 3;
	i23[0] = 2; i23[1] = 3;
	i32[0] = 3; i32[1] = 2;

	se_part<2, double> sp1(bis, m11, 2), sp2(bis, m11, 4);
	sp1.add_map(i00, i11, true);
	sp1.add_map(i01, i10, true);
	sp2.add_map(i00, i11, true);
	sp2.add_map(i01, i10, true);
	sp2.add_map(i22, i33, true);
	sp2.add_map(i23, i32, true);

	symmetry_element_set<2, double> set1(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se_part<2, double>::k_sym_type);
	set1.insert(sp1);
	set1.insert(sp2);
	set2_ref.insert(sp1);
	set2_ref.insert(sp2);

	partition_set<2, double> pset(set1);
	pset.convert(set2);

	compare_ref<2>::compare(testname, bis, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Adding one partition to an empty partition set
 **/
void partition_set_test::test_add_1(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_add_1(bool)";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	index<2> i00, i01, i10, i11;
	i01[1] = 1; i10[0] = 1;
	i11[0] = 1; i11[1] = 1;

	se_part<2, double> sp(bis, m11, 2);
	sp.add_map(i00, i11, sign);
	sp.add_map(i01, i10, sign);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set_ref(se_part<2, double>::k_sym_type);
	set_ref.insert(sp);

	permutation<2> perm;
	partition_set<2, double> pset(bis);
	pset.add_partition(sp, perm);
	pset.convert(set);

	compare_ref<2>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Adding one partition to an empty partition set with permutation
 **/
void partition_set_test::test_add_2(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_add_2(bool)";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1111, 2);
	bis.split(m1111, 5);
	bis.split(m1111, 7);

	index<4> i0000, i0101, i1010, i0110, i1001, i1111;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se_part<4, double> sp1(bis, m1111, 2), sp2(bis, m1111, 2);
	sp1.add_map(i0000, i1111, sign);
	sp1.add_map(i0101, i1010, sign);
	sp2.add_map(i0000, i1111, sign);
	sp2.add_map(i0110, i1001, sign);

	symmetry_element_set<4, double> set(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set_ref(se_part<4, double>::k_sym_type);
	set_ref.insert(sp2);

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);
	partition_set<4, double> pset(bis);
	pset.add_partition(sp1, perm);
	pset.convert(set);

	compare_ref<4>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Adding a partition to a non-empty partition set with permutation
		(all dimensions partitioned)
 **/
void partition_set_test::test_add_3a(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_add_3a(bool)";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1111, 2);
	bis.split(m1111, 5);
	bis.split(m1111, 7);

	index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
		i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se_part<4, double> sp1(bis, m1111, 2), sp2(bis, m1111, 2);
	sp1.add_map(i0000, i1111, sign);
	sp1.add_map(i0101, i1010, sign);
	sp2.add_map(i0011, i1100, sign);
	sp2.add_map(i0110, i1001, sign);

	se_part<4, double> sp_ref(bis, m1111, 2);
	sp_ref.add_map(i0000, i1111, sign);
	sp_ref.add_map(i0101, i1010, sign);
	sp_ref.add_map(i0011, i1100, sign);

	symmetry_element_set<4, double> set1(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set2(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se_part<4, double>::k_sym_type);
	set1.insert(sp1);
	set2_ref.insert(sp_ref);

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);
	partition_set<4, double> pset(set1);
	pset.add_partition(sp2, perm);
	pset.convert(set2);

	compare_ref<4>::compare(testname, bis, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Adding a partition to a non-empty partition set with permutation
		(all dimensions partitioned). Wrong sign!
 **/
void partition_set_test::test_add_3b(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_add_3b(bool)";

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1111, 2);
	bis.split(m1111, 5);
	bis.split(m1111, 7);

	index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
		i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se_part<4, double> sp1(bis, m1111, 2), sp2(bis, m1111, 2);
	sp1.add_map(i0000, i1111, sign);
	sp1.add_map(i0101, i1010, sign);
	sp2.add_map(i0011, i1100, !sign);
	sp2.add_map(i0110, i1001, !sign);

	symmetry_element_set<4, double> set1(se_part<4, double>::k_sym_type);
	set1.insert(sp1);

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);

	bool found = false;
	try {

	partition_set<4, double> pset(set1);
	pset.add_partition(sp2, perm);

	} catch (exception &e) {
		found = true;
	}

	if (! found) {
		fail_test(testname, __FILE__, __LINE__, "No exception.");
	}
}

/**	\test Adding a partition to a non-empty partition set with permutation
		[0123->1320] (disjoint partitioned dimensions)
 **/
void partition_set_test::test_add_4(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_add_4()";

	try {

	index<4> i1, i2;
	i2[0] = 15; i2[1] = 9; i2[2] = 9; i2[3] = 15;
	block_index_space<4> bisa(dimensions<4>(index_range<4>(i1, i2)));

	mask<4> m1001, m0110, m1010, m1100, m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	m1001[0] = true; m0110[1] = true; m0110[2] = true; m1001[3] = true;
	m1010[0] = true; m1010[2] = true;
	m1100[0] = true; m1100[1] = true;
	bisa.split(m0110, 2);
	bisa.split(m0110, 5);
	bisa.split(m0110, 7);
	bisa.split(m1001, 4);
	bisa.split(m1001, 8);
	bisa.split(m1001, 12);

	permutation<4> perm; perm.permute(0, 1).permute(1, 3);

	block_index_space<4> bisb(bisa);
	bisb.permute(perm);

	index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
		i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se_part<4, double> sp1(bisb, m1100, 2), sp2(bisa, m1010, 2);
	sp1.add_map(i0000, i1100, sign);
	sp1.add_map(i0100, i1000, sign);
	sp2.add_map(i0000, i1010, sign);
	sp2.add_map(i0010, i1000, sign);

	se_part<4, double> sp_ref(bisb, m1111, 2);
	sp_ref.add_map(i0000, i1100, sign);
	sp_ref.add_map(i0001, i1101, sign);
	sp_ref.add_map(i0010, i1110, sign);
	sp_ref.add_map(i0011, i1111, sign);
	sp_ref.add_map(i0100, i1000, sign);
	sp_ref.add_map(i0101, i1001, sign);
	sp_ref.add_map(i0110, i1010, sign);
	sp_ref.add_map(i0111, i1011, sign);

	sp_ref.add_map(i0000, i0011, sign);
	sp_ref.add_map(i0100, i0111, sign);
	sp_ref.add_map(i1000, i1011, sign);
	sp_ref.add_map(i1100, i1111, sign);
	sp_ref.add_map(i0001, i0010, sign);
	sp_ref.add_map(i0101, i0110, sign);
	sp_ref.add_map(i1001, i1010, sign);
	sp_ref.add_map(i1101, i1110, sign);

	symmetry_element_set<4, double> set1(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set2(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se_part<4, double>::k_sym_type);
	set1.insert(sp1);
	set2_ref.insert(sp_ref);

	partition_set<4, double> pset(set1);
	pset.add_partition(sp2, perm);
	pset.convert(set2);

	compare_ref<4>::compare(testname, bisb, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Adding a partition to a non-empty partition set with permutation
		[0123->1203] (overlapping partitioned dimensions)
 **/
void partition_set_test::test_add_5() throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_add_5()";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 15; i2[3] = 15;
	block_index_space<4> bisa(dimensions<4>(index_range<4>(i1, i2)));

	mask<4> m0011, m1100, m1010, m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
	m1010[0] = true; m1010[2] = true;
	bisa.split(m1100, 2); bisa.split(m1100, 5); bisa.split(m1100, 7);
	bisa.split(m0011, 4); bisa.split(m0011, 8); bisa.split(m0011, 12);

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);

	block_index_space<4> bisb(bisa);
	bisb.permute(perm);

	index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
		i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se_part<4, double> sp1(bisb, m1100, 2), sp2(bisa, m1010, 2);
	sp1.add_map(i0000, i1100, true);
	sp1.add_map(i0100, i1000, true);
	sp2.add_map(i0000, i1010, true);
	sp2.add_map(i0010, i1000, true);

	se_part<4, double> sp_ref(bisb, m1111, 2);
	sp_ref.add_map(i0000, i1100, true);
	sp_ref.add_map(i0001, i1101, true);
	sp_ref.add_map(i0010, i1110, true);
	sp_ref.add_map(i0011, i1111, true);

	sp_ref.add_map(i0100, i1000, true);
	sp_ref.add_map(i0101, i1001, true);
	sp_ref.add_map(i0110, i1010, true);
	sp_ref.add_map(i0111, i1011, true);

	sp_ref.add_map(i0000, i0110, true);
	sp_ref.add_map(i0001, i0111, true);
	sp_ref.add_map(i1000, i1110, true);
	sp_ref.add_map(i1001, i1111, true);

	sp_ref.add_map(i0010, i0100, true);
	sp_ref.add_map(i0011, i0101, true);
	sp_ref.add_map(i1010, i1100, true);
	sp_ref.add_map(i1011, i1101, true);

	symmetry_element_set<4, double> set1(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set2(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se_part<4, double>::k_sym_type);
	set1.insert(sp1);
	set2_ref.insert(sp_ref);

	partition_set<4, double> pset(set1);
	pset.add_partition(sp2, perm);
	pset.convert(set2);

	compare_ref<4>::compare(testname, bisb, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Adding a partition of less dimensions to an empty partition set
 **/
void partition_set_test::test_add_6(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_add_6(bool)";

	try {

	index<1> i1a, i2a;
	i2a[0] = 9;
	block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
	mask<1> m1;
	m1[0] = true;
	bisa.split(m1, 2);
	bisa.split(m1, 5);
	bisa.split(m1, 7);

	index<2> i1b, i2b;
	i2b[0] = 9; i2b[1] = 9;
	block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));
	mask<2> m01, m10, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bisb.split(m11, 2);
	bisb.split(m11, 5);
	bisb.split(m11, 7);

	index<1> i0, i1;
	index<2> i00, i01, i10, i11;
	i0[0] = 0; i1[0] = 1;
	i01[1] = 1; i10[0] = 1;
	i11[0] = 1; i11[1] = 1;

	se_part<1, double> sp1(bisa, m1, 2);
	sp1.add_map(i0, i1, sign);

	se_part<2, double> sp2(bisb, m11, 2);
	sp2.add_map(i00, i01, sign);
	sp2.add_map(i10, i11, sign);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set_ref(se_part<2, double>::k_sym_type);
	set_ref.insert(sp2);

	permutation<1> perm;
	partition_set<2, double> pset(bisb);
	pset.add_partition(sp1, perm, m01);
	pset.convert(set);

	compare_ref<2>::compare(testname, bisb, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Adding a partition of less dimensions to a non-empty partition set
 **/
void partition_set_test::test_add_7(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_add_7(bool)";

	try {

	index<1> i1a, i2a;
	i2a[0] = 9;
	block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
	mask<1> m1;
	m1[0] = true;
	bisa.split(m1, 2);
	bisa.split(m1, 5);
	bisa.split(m1, 7);

	index<2> i1b, i2b;
	i2b[0] = 9; i2b[1] = 9;
	block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));
	mask<2> m01, m10, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bisb.split(m11, 2);
	bisb.split(m11, 5);
	bisb.split(m11, 7);

	index<1> i0, i1;
	index<2> i00, i01, i10, i11;
	i0[0] = 0; i1[0] = 1;
	i01[1] = 1; i10[0] = 1;
	i11[0] = 1; i11[1] = 1;

	se_part<1, double> sp1(bisa, m1, 2);
	sp1.add_map(i0, i1, sign);

	se_part<2, double> sp2(bisb, m11, 2), sp_ref(bisb, m11, 2);
	sp2.add_map(i00, i11, true);
	sp_ref.add_map(i00, i01, sign);
	sp_ref.add_map(i10, i11, sign);
	sp_ref.add_map(i00, i11, true);

	symmetry_element_set<2, double> set1(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se_part<2, double>::k_sym_type);
	set1.insert(sp2);
	set2_ref.insert(sp_ref);

	permutation<1> perm;
	partition_set<2, double> pset(set1);
	pset.add_partition(sp1, perm, m01);
	pset.convert(set2);

	compare_ref<2>::compare(testname, bisb, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Adding a partition of less dimensions to a non-empty partition set
		with permutation.
 **/
void partition_set_test::test_add_8(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_add_8(bool)";

	try {

	index<2> i1a, i2a;
	i2a[0] = 9; i2a[1] = 19;
	block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
	mask<2> m01, m10, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bisa.split(m10, 2);
	bisa.split(m10, 5);
	bisa.split(m10, 7);
	bisa.split(m01, 5);
	bisa.split(m01, 10);
	bisa.split(m01, 15);

	index<3> i1b, i2b;
	i2b[0] = 19; i2b[1] = 9; i2b[2] = 9;
	block_index_space<3> bisb(dimensions<3>(index_range<3>(i1b, i2b)));
	mask<3> m011, m100, m111, m010, m101;
	m101[0] = true; m010[1] = true; m101[2] = true;
	m100[0] = true; m011[1] = true; m011[2] = true;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bisb.split(m011, 2);
	bisb.split(m011, 5);
	bisb.split(m011, 7);
	bisb.split(m100, 5);
	bisb.split(m100, 10);
	bisb.split(m100, 15);

	index<2> i00, i01, i10, i11;
	index<3> i000, i111, i001, i110, i010, i101, i011, i100;
	i01[1] = 1; i10[0] = 1;
	i11[0] = 1; i11[1] = 1;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i100[0] = 1; i011[1] = 1; i011[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<2, double> sp1(bisa, m11, 2);
	sp1.add_map(i00, i11, sign);
	sp1.add_map(i01, i10, sign);

	se_part<3, double> sp2(bisb, m010, 2), sp_ref(bisb, m111, 2);
	sp2.add_map(i000, i010, sign);
	sp_ref.add_map(i000, i101, sign);
	sp_ref.add_map(i010, i111, sign);
	sp_ref.add_map(i001, i100, sign);
	sp_ref.add_map(i011, i110, sign);
	sp_ref.add_map(i000, i010, sign);
	sp_ref.add_map(i001, i011, sign);
	sp_ref.add_map(i100, i110, sign);
	sp_ref.add_map(i101, i111, sign);

	symmetry_element_set<3, double> set1(se_part<3, double>::k_sym_type);
	symmetry_element_set<3, double> set2(se_part<3, double>::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se_part<3, double>::k_sym_type);
	set1.insert(sp2);
	set2_ref.insert(sp_ref);

	permutation<2> perm; perm.permute(0, 1);
	partition_set<3, double> pset(set1);
	pset.add_partition(sp1, perm, m101);
	pset.convert(set2);

	compare_ref<3>::compare(testname, bisb, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Permuting an empty partition set.
 **/
void partition_set_test::test_permute_1() throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_permute_1()";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 15; i2[3] = 15;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m0011, m1100;
	m1100[0] = true; m1100[1] = true;
	m0011[2] = true; m0011[3] = true;
	bis.split(m1100, 2);
	bis.split(m1100, 5);
	bis.split(m1100, 7);
	bis.split(m0011, 4);
	bis.split(m0011, 8);
	bis.split(m0011, 12);

	symmetry_element_set<4, double> set(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set_ref(se_part<4, double>::k_sym_type);

	partition_set<4, double> pset(bis);

	permutation<4> perm; perm.permute(0, 1).permute(1, 2);
	pset.permute(perm);
	pset.convert(set);

	bis.permute(perm);

	compare_ref<4>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Permuting a partition set with one partition.
 **/
void partition_set_test::test_permute_2(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_permute_2(bool)";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 15; i2[3] = 15;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m0011, m1100, m1111;
	m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1100, 2);
	bis.split(m1100, 5);
	bis.split(m1100, 7);
	bis.split(m0011, 4);
	bis.split(m0011, 8);
	bis.split(m0011, 12);

	index<4> i0000, i1111, i0011, i1100, i0101, i1010;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se_part<4, double> sp(bis, m1111, 2);
	sp.add_map(i0000, i1111, sign);
	sp.add_map(i0011, i1100, sign);

	permutation<4> p0;
	partition_set<4, double> pset(bis);
	pset.add_partition(sp, p0);

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);
	pset.permute(perm);

	symmetry_element_set<4, double> set(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set_ref(se_part<4, double>::k_sym_type);

	pset.convert(set);

	bis.permute(perm);
	se_part<4, double> sp_ref(bis, m1111, 2);
	sp_ref.add_map(i0000, i1111, sign);
	sp_ref.add_map(i0101, i1010, sign);
	set_ref.insert(sp_ref);

	compare_ref<4>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Permuting a partition set with a full and a partial partition.
 **/
void partition_set_test::test_permute_3(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_permute_3(bool)";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 15; i2[3] = 15;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m0011, m0101, m1100, m1111;
	m0101[1] = true; m0101[3] = true;
	m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1100, 2);
	bis.split(m1100, 5);
	bis.split(m1100, 7);
	bis.split(m0011, 4);
	bis.split(m0011, 8);
	bis.split(m0011, 12);

	index<4> i0000, i1111, i0011, i1100, i0101, i1010;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
	index<4> i0022, i0202, i0033, i0303;
	i0022[2] = 2; i0022[3] = 2; i0202[1] = 2; i0202[3] = 2;
	i0033[2] = 3; i0033[3] = 3; i0303[1] = 3; i0303[3] = 3;

	se_part<4, double> sp1(bis, m1111, 2), sp2(bis, m0011, 4);
	sp1.add_map(i0000, i1111, sign);
	sp1.add_map(i0011, i1100, sign);
	sp2.add_map(i0000, i0011, sign);
	sp2.add_map(i0022, i0033, sign);

	permutation<4> p0;
	partition_set<4, double> pset(bis);
	pset.add_partition(sp1, p0);
	pset.add_partition(sp2, p0);

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);
	pset.permute(perm);

	symmetry_element_set<4, double> set(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set_ref(se_part<4, double>::k_sym_type);

	pset.convert(set);

	bis.permute(perm);
	se_part<4, double> sp1_ref(bis, m1111, 2), sp2_ref(bis, m0101, 4);
	sp1_ref.add_map(i0000, i1111, sign);
	sp1_ref.add_map(i0101, i1010, sign);
	sp2_ref.add_map(i0000, i0101, sign);
	sp2_ref.add_map(i0202, i0303, sign);
	set_ref.insert(sp1_ref);
	set_ref.insert(sp2_ref);

	compare_ref<4>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Permuting a partition set with a full partition.
 **/
void partition_set_test::test_permute_4(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_permute_4(bool)";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1111, 2);
	bis.split(m1111, 5);
	bis.split(m1111, 7);

	index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
		i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se_part<4, double> sp(bis, m1111, 2), sp_ref(bis, m1111, 2);
	sp.add_map(i0000, i0101, sign);
	sp.add_map(i0101, i1010, sign);
	sp.add_map(i1010, i1111, sign);
	sp.add_map(i0010, i0111, sign);
	sp.add_map(i0111, i1000, sign);
	sp.add_map(i1000, i1101, sign);
	sp.add_map(i0011, i0110, sign);
	sp.add_map(i0110, i1001, sign);
	sp.add_map(i1001, i1100, sign);
	sp.add_map(i0001, i0100, sign);
	sp.add_map(i0100, i1011, sign);
	sp.add_map(i1011, i1110, sign);

	sp_ref.add_map(i0000, i0110, sign);
	sp_ref.add_map(i0110, i1001, sign);
	sp_ref.add_map(i1001, i1111, sign);
	sp_ref.add_map(i0010, i1011, sign);
	sp_ref.add_map(i1011, i0100, sign);
	sp_ref.add_map(i0100, i1101, sign);
	sp_ref.add_map(i0011, i1010, sign);
	sp_ref.add_map(i1010, i0101, sign);
	sp_ref.add_map(i0101, i1100, sign);
	sp_ref.add_map(i0001, i1000, sign);
	sp_ref.add_map(i1000, i0111, sign);
	sp_ref.add_map(i0111, i1110, sign);

	permutation<4> p0;
	partition_set<4, double> pset(bis);
	pset.add_partition(sp, p0);

	permutation<4> perm;
	perm.permute(0, 1);
	pset.permute(perm);

	symmetry_element_set<4, double> set(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set_ref(se_part<4, double>::k_sym_type);

	pset.convert(set);

	set_ref.insert(sp_ref);

	compare_ref<4>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two empty partition sets
 **/
void partition_set_test::test_intersect_1() throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_1()";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	partition_set<2, double> pset1(bis), pset2(bis);
	pset1.intersect(pset2);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);
	pset1.convert(set);

	if (! set.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of an empty partition set with a non-empty one.
 **/
void partition_set_test::test_intersect_2(bool first_empty)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_2(bool)";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	se_part<2, double> sp(bis, m11, 2);
	index<2> i00, i11;
	sp.add_map(i00, i11, true);

	partition_set<2, double> pset1(bis), pset2(bis);
	permutation<2> p0;
	if (first_empty)
		pset2.add_partition(sp, p0);
	else
		pset1.add_partition(sp, p0);

	pset1.intersect(pset2);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);
	pset1.convert(set);

	if (! set.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Intersection of two non-empty, but disjoint partition sets
 **/
void partition_set_test::test_intersect_3(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_3(bool)";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m10, m01, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	se_part<2, double> sp1(bis, m01, 2), sp2(bis, m10, 2);
	sp1.add_map(i00, i01, sign);
	sp2.add_map(i00, i10, sign);

	partition_set<2, double> pset1(bis), pset2(bis);
	permutation<2> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp2, p0);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);

	pset2.intersect(pset1);
	pset2.convert(set);

	if (! set.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two identical, non-empty partition sets
 **/
void partition_set_test::test_intersect_4(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_4(bool)";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m10, m01, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	se_part<2, double> sp1(bis, m11, 2), sp2(bis, m11, 2);
	sp1.add_map(i00, i11, true);
	sp1.add_map(i01, i10, true);

	partition_set<2, double> pset1(bis), pset2(bis);
	permutation<2> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp1, p0);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set_ref(se_part<2, double>::k_sym_type);
	set_ref.insert(sp1);

	pset2.intersect(pset1);
	pset2.convert(set);

	compare_ref<2>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two non-empty and overlapping partition sets (! mult)
 **/
void partition_set_test::test_intersect_5a(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_5a(bool)";

	try {

	index<3> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 5);
	bis.split(m111, 7);

	index<3> i000, i001, i110, i010, i101, i111;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<3, double> sp1(bis, m111, 2), sp2(bis, m111, 2),
			sp_ref(bis, m111, 2);
	sp1.add_map(i000, i111, sign);
	sp1.add_map(i001, i110, sign);
	sp2.add_map(i000, i111, sign);
	sp2.add_map(i010, i101, sign);
	sp_ref.add_map(i000, i111, sign);

	partition_set<3, double> pset1(bis), pset2(bis);
	permutation<3> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp2, p0);

	symmetry_element_set<3, double> set(se_part<3, double>::k_sym_type);
	symmetry_element_set<3, double> set_ref(se_part<3, double>::k_sym_type);
	set_ref.insert(sp_ref);

	pset2.intersect(pset1, false);
	pset2.convert(set);

	compare_ref<3>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two non-empty and overlapping partition sets (mult)
 **/
void partition_set_test::test_intersect_5b(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_5b(bool)";

	try {

	index<3> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 5);
	bis.split(m111, 7);

	index<3> i000, i001, i110, i010, i101, i111;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<3, double> sp1(bis, m111, 2), sp2(bis, m111, 2),
			sp_ref(bis, m111, 2);
	sp1.add_map(i000, i111, sign);
	sp1.add_map(i001, i110, sign);
	sp2.add_map(i000, i111, sign);
	sp2.add_map(i010, i101, sign);
	sp_ref.add_map(i000, i111, true);

	partition_set<3, double> pset1(bis), pset2(bis);
	permutation<3> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp2, p0);

	symmetry_element_set<3, double> set(se_part<3, double>::k_sym_type);
	symmetry_element_set<3, double> set_ref(se_part<3, double>::k_sym_type);
	set_ref.insert(sp_ref);

	pset2.intersect(pset1, true);
	pset2.convert(set);

	compare_ref<3>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two non-empty, overlapping partition sets
		(opposite sign, ! mult)
 **/
void partition_set_test::test_intersect_6a(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_6a()";

	try {

	index<3> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 5);
	bis.split(m111, 7);

	index<3> i000, i001, i110, i010, i101, i111;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<3, double> sp1(bis, m111, 2), sp2(bis, m111, 2);
	sp1.add_map(i000, i111, sign);
	sp1.add_map(i001, i110, sign);
	sp2.add_map(i000, i111, ! sign);
	sp2.add_map(i010, i101, ! sign);

	partition_set<3, double> pset1(bis), pset2(bis);
	permutation<3> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp2, p0);

	symmetry_element_set<3, double> set(se_part<3, double>::k_sym_type);

	pset2.intersect(pset1, false);
	pset2.convert(set);

	if (! set.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two non-empty, overlapping partition sets
		(opposite sign, mult)
 **/
void partition_set_test::test_intersect_6b(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_6b(bool)";

	try {

	index<3> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 5);
	bis.split(m111, 7);

	index<3> i000, i001, i110, i010, i101, i111;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<3, double> sp1(bis, m111, 2), sp2(bis, m111, 2),
			sp_ref(bis, m111, 2);
	sp1.add_map(i000, i111, sign);
	sp1.add_map(i001, i110, sign);
	sp2.add_map(i000, i111, ! sign);
	sp2.add_map(i010, i101, ! sign);
	sp_ref.add_map(i000, i111, false);

	partition_set<3, double> pset1(bis), pset2(bis);
	permutation<3> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp2, p0);

	symmetry_element_set<3, double> set(se_part<3, double>::k_sym_type);
	symmetry_element_set<3, double> set_ref(se_part<3, double>::k_sym_type);
	set_ref.insert(sp_ref);

	pset2.intersect(pset1, true);
	pset2.convert(set);

	compare_ref<3>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two non-empty, overlapping partition sets. Overlap
		is not accessible by direct map in first partition set
 **/
void partition_set_test::test_intersect_7a(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_7a(bool)";

	try {

	index<3> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 5);
	bis.split(m111, 7);

	index<3> i000, i111, i001, i110, i010, i101, i011, i100;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i100[0] = 1; i011[1] = 1; i011[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<3, double> sp1(bis, m111, 2), sp2(bis, m111, 2),
			sp_ref(bis, m111, 2);
	sp1.add_map(i000, i001, sign);
	sp1.add_map(i001, i010, sign);
	sp1.add_map(i010, i011, sign);
	sp1.add_map(i011, i100, sign);
	sp1.add_map(i100, i101, sign);
	sp1.add_map(i101, i110, sign);
	sp1.add_map(i110, i111, sign);
	sp2.add_map(i001, i110, sign);
	sp_ref.add_map(i001, i110, sign);

	partition_set<3, double> pset1(bis), pset2(bis);
	permutation<3> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp2, p0);

	symmetry_element_set<3, double> set(se_part<3, double>::k_sym_type);
	symmetry_element_set<3, double> set_ref(se_part<3, double>::k_sym_type);
	set_ref.insert(sp_ref);

	pset2.intersect(pset1);
	pset2.convert(set);

	compare_ref<3>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two non-empty, overlapping partition sets. Overlap
		is not accessible by direct map in second partition set
 **/
void partition_set_test::test_intersect_7b(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_7b(bool)";

	try {

	index<3> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 5);
	bis.split(m111, 7);

	index<3> i000, i111, i001, i110, i010, i101, i011, i100;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i100[0] = 1; i011[1] = 1; i011[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<3, double> sp1(bis, m111, 2), sp2(bis, m111, 2),
			sp_ref(bis, m111, 2);
	sp1.add_map(i001, i110, sign);
	sp2.add_map(i000, i001, sign);
	sp2.add_map(i001, i010, sign);
	sp2.add_map(i010, i011, sign);
	sp2.add_map(i011, i100, sign);
	sp2.add_map(i100, i101, sign);
	sp2.add_map(i101, i110, sign);
	sp2.add_map(i110, i111, sign);
	sp_ref.add_map(i001, i110, sign);

	partition_set<3, double> pset1(bis), pset2(bis);
	permutation<3> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp2, p0);

	symmetry_element_set<3, double> set(se_part<3, double>::k_sym_type);
	symmetry_element_set<3, double> set_ref(se_part<3, double>::k_sym_type);
	set_ref.insert(sp_ref);

	pset2.intersect(pset1);
	pset2.convert(set);

	compare_ref<3>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two non-empty, overlapping partition sets. Overlap
		is not accessible by direct maps
 **/
void partition_set_test::test_intersect_7c(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_7c(bool)";

	try {

	index<3> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 5);
	bis.split(m111, 7);

	index<3> i000, i111, i001, i110, i010, i101, i011, i100;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i100[0] = 1; i011[1] = 1; i011[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<3, double> sp1(bis, m111, 2), sp2(bis, m111, 2),
			sp_ref(bis, m111, 2);
	sp1.add_map(i001, i100, sign);
	sp1.add_map(i100, i101, sign);
	sp1.add_map(i101, i110, sign);
	sp1.add_map(i110, i111, sign);
	sp2.add_map(i000, i001, sign);
	sp2.add_map(i001, i010, sign);
	sp2.add_map(i010, i011, sign);
	sp2.add_map(i011, i110, sign);
	sp_ref.add_map(i001, i110, sign);

	partition_set<3, double> pset1(bis), pset2(bis);
	permutation<3> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp2, p0);

	symmetry_element_set<3, double> set(se_part<3, double>::k_sym_type);
	symmetry_element_set<3, double> set_ref(se_part<3, double>::k_sym_type);
	set_ref.insert(sp_ref);

	pset2.intersect(pset1);
	pset2.convert(set);

	compare_ref<3>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two non-empty, overlapping partition sets. Overlap
		is not accessible by direct maps
 **/
void partition_set_test::test_intersect_8(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_intersect_8(bool)";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1111, 2);
	bis.split(m1111, 5);
	bis.split(m1111, 7);

	index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
		i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se_part<4, double> sp1(bis, m1111, 2), sp2(bis, m1111, 2),
			sp_ref(bis, m1111, 2);
	sp1.add_map(i0000, i0101, sign);
	sp1.add_map(i0101, i1010, sign);
	sp1.add_map(i1010, i1111, sign);
	sp1.add_map(i0010, i0111, sign);
	sp1.add_map(i0111, i1000, sign);
	sp1.add_map(i1000, i1101, sign);
	sp1.add_map(i0011, i0110, sign);
	sp1.add_map(i0110, i1001, sign);
	sp1.add_map(i1001, i1100, sign);
	sp1.add_map(i0001, i0100, sign);
	sp1.add_map(i0100, i1011, sign);
	sp1.add_map(i1011, i1110, sign);

	sp2.add_map(i0000, i0110, sign);
	sp2.add_map(i0110, i1001, sign);
	sp2.add_map(i1001, i1111, sign);
	sp2.add_map(i0010, i1011, sign);
	sp2.add_map(i1011, i0100, sign);
	sp2.add_map(i0100, i1101, sign);
	sp2.add_map(i0011, i1010, sign);
	sp2.add_map(i1010, i0101, sign);
	sp2.add_map(i0101, i1100, sign);
	sp2.add_map(i0001, i1000, sign);
	sp2.add_map(i1000, i0111, sign);
	sp2.add_map(i0111, i1110, sign);

	sp_ref.add_map(i0000, i1111, sign);
	sp_ref.add_map(i0101, i1010, sign);
	sp_ref.add_map(i0010, i1101, sign);
	sp_ref.add_map(i0111, i1000, sign);
	sp_ref.add_map(i0011, i1100, sign);
	sp_ref.add_map(i0110, i1001, sign);
	sp_ref.add_map(i0001, i1110, sign);
	sp_ref.add_map(i0100, i1011, sign);

	partition_set<4, double> pset1(bis), pset2(bis);
	permutation<4> p0;
	pset1.add_partition(sp1, p0);
	pset2.add_partition(sp2, p0);

	symmetry_element_set<4, double> set(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set_ref(se_part<4, double>::k_sym_type);
	set_ref.insert(sp_ref);

	pset2.intersect(pset1);
	pset2.convert(set);

	compare_ref<4>::compare(testname, bis, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Stabilize an empty partition set
 **/
void partition_set_test::test_stabilize_1()
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_stabilize_1()";

	try {

	index<2> i1a, i2a;
	i2a[0] = 9; i2a[1] = 9;
	block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));

	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bisa.split(m11, 2);
	bisa.split(m11, 5);
	bisa.split(m11, 7);

	index<4> i1b, i2b;
	i2b[0] = 9; i2b[1] = 9; i2b[2] = 9; i2b[3] = 9;
	block_index_space<4> bisb(dimensions<4>(index_range<4>(i1b, i2b)));


	mask<4> m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bisb.split(m1111, 2);
	bisb.split(m1111, 5);
	bisb.split(m1111, 7);

	partition_set<4, double> pset1(bisb);
	partition_set<2, double> pset2(bisa);

	mask<4> msk[1];
	msk[0][2] = true; msk[0][3] = true;
	pset1.stabilize(msk, pset2);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);

	pset2.convert(set);

	if (! set.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Stabilize a partition set with a single partition, one stabilizing step
 **/
void partition_set_test::test_stabilize_2(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_stabilize_2(bool)";

	try {

	index<2> i1a, i2a;
	i2a[0] = 9; i2a[1] = 9;
	block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));

	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bisa.split(m11, 2);
	bisa.split(m11, 5);
	bisa.split(m11, 7);

	index<4> i1b, i2b;
	i2b[0] = 9; i2b[1] = 9; i2b[2] = 9; i2b[3] = 9;
	block_index_space<4> bisb(dimensions<4>(index_range<4>(i1b, i2b)));

	mask<4> m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bisb.split(m1111, 2);
	bisb.split(m1111, 5);
	bisb.split(m1111, 7);

	se_part<4, double> sp1(bisb, m1111, 2);
	index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
		i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	sp1.add_map(i0000, i1010, sign);
	sp1.add_map(i0001, i1011, sign);
	sp1.add_map(i0100, i1110, sign);
	sp1.add_map(i0101, i1111, sign);

	sp1.add_map(i0000, i0101, sign);
	sp1.add_map(i0010, i0111, sign);
	sp1.add_map(i1000, i1101, sign);
	sp1.add_map(i1010, i1111, sign);

	sp1.add_map(i0010, i1000, sign);
	sp1.add_map(i0011, i1001, sign);
	sp1.add_map(i0110, i1100, sign);
	sp1.add_map(i0111, i1101, sign);

	sp1.add_map(i0001, i0100, sign);
	sp1.add_map(i0011, i0110, sign);
	sp1.add_map(i1001, i1100, sign);
	sp1.add_map(i1011, i1110, sign);

	se_part<2, double> sp2(bisa, m11, 2);
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	sp2.add_map(i00, i11, sign);
	sp2.add_map(i01, i10, sign);

	partition_set<4, double> pset1(bisb);
	pset1.add_partition(sp1, permutation<4>());

	partition_set<2, double> pset2(bisa);

	mask<4> msk[1];
	msk[0][2] = true; msk[0][3] = true;
	pset1.stabilize(msk, pset2);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set_ref(se_part<2, double>::k_sym_type);
	set_ref.insert(sp2);

	pset2.convert(set);

	compare_ref<2>::compare(testname, bisa, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Stabilize a partition set with a single partition, two partition steps
 **/
void partition_set_test::test_stabilize_3(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_stabilize_3(bool)";

	try {

	index<2> i1a, i2a;
	i2a[0] = 9; i2a[1] = 15;
	block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));

	mask<2> m01, m10, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bisa.split(m10, 2);
	bisa.split(m10, 5);
	bisa.split(m10, 7);
	bisa.split(m01, 4);
	bisa.split(m01, 8);
	bisa.split(m01, 12);

	index<6> i1b, i2b;
	i2b[0] = 9; i2b[1] = 15; i2b[2] = 9; i2b[3] = 9; i2b[4] = 15; i2b[5] = 15;
	block_index_space<6> bisb(dimensions<6>(index_range<6>(i1b, i2b)));

	mask<6> m010011, m101100, m000101, m111010, m111111;
	m101100[0] = true; m010011[1] = true; m101100[2] = true;
	m101100[3] = true; m010011[4] = true; m010011[5] = true;
	m111111[0] = true; m111111[1] = true; m111111[2] = true;
	m111111[3] = true; m111111[4] = true; m111111[5] = true;
	m111010[0] = true; m111010[1] = true; m111010[2] = true;
	m000101[3] = true; m111010[4] = true; m000101[5] = true;
	bisb.split(m101100, 2);
	bisb.split(m101100, 5);
	bisb.split(m101100, 7);
	bisb.split(m010011, 4);
	bisb.split(m010011, 8);
	bisb.split(m010011, 12);

	se_part<6, double> sp1(bisb, m111111, 2);

	index<2> ix1, ix2;
	ix1[0] = 1; ix2[1] = 1;
	dimensions<2> pdimsx(index_range<2>(ix1, ix2));
	index<4> iy1, iy2;
	iy2[0] = 1; iy2[1] = 1; iy2[2] = 1; iy2[3] = 1;
	dimensions<4> pdimsy(index_range<4>(iy1, iy2));

	abs_index<4> ai1(pdimsy);
	while (ai1.get_abs_index() < pdimsy.get_size() / 2) {
		abs_index<2> aj(pdimsx);
		do {
			index<6> i1, i2;
			for (size_t i = 0, j = 0, k = 0; i < 6; i++) {
				if (m111010[i]) {
					i1[i] = ai1.get_index()[j];
					i2[i] = (ai1.get_index()[j] + 1) % 2;
					j++;
				}
				else {
					i1[i] = i2[i] = aj.get_index()[k++];
				}
			}
			sp1.add_map(i1, i2, true);
		} while (aj.inc());
		ai1.inc();
	}

	abs_index<2> ai2(pdimsx);
	while (ai2.get_abs_index() < pdimsx.get_size() / 2) {
		abs_index<4> aj(pdimsy);
		do {
			index<6> i1, i2;
			for (size_t i = 0, j = 0, k = 0; i < 6; i++) {
				if (m000101[i]) {
					i1[i] = ai2.get_index()[j];
					i2[i] = (ai2.get_index()[j] + 1) % 2;
					j++;
				}
				else {
					i1[i] = i2[i] = aj.get_index()[k++];
				}
			}
			sp1.add_map(i1, i2, sign);
		} while (aj.inc());
		ai2.inc();
	}

	se_part<2, double> sp2(bisa, m11, 2);
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	sp2.add_map(i00, i11, sign);
	sp2.add_map(i01, i10, sign);

	partition_set<6, double> pset1(bisb);
	pset1.add_partition(sp1, permutation<6>());

	partition_set<2, double> pset2(bisa);

	mask<6> msk[2];
	msk[0][4] = true; msk[0][5] = true;
	msk[1][2] = true; msk[1][3] = true;
	pset1.stabilize(msk, pset2);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set_ref(se_part<2, double>::k_sym_type);
	set_ref.insert(sp2);

	pset2.convert(set);

	compare_ref<2>::compare(testname, bisa, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Stabilize several dimensions of a partition sets
 **/
void partition_set_test::test_stabilize_4(bool sign)
	throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_stabilize_4(bool)";

	try {

	index<2> i1a, i2a, i1b, i2b;
	i2a[0] = 9; i2a[1] = 19;
	i2b[0] = 19; i2b[1] = 19;
	block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
	block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));

	mask<2> m10, m01, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bisa.split(m10, 2);
	bisa.split(m10, 5);
	bisa.split(m10, 7);
	bisa.split(m01, 3);
	bisa.split(m01, 6);
	bisa.split(m01, 10);
	bisa.split(m01, 13);
	bisa.split(m01, 16);
	bisb.split(m11, 3);
	bisb.split(m11, 6);
	bisb.split(m11, 10);
	bisb.split(m11, 13);
	bisb.split(m11, 16);

	index<4> i1c, i2c;
	i2c[0] = 9; i2c[1] = 19; i2c[2] = 19; i2c[3] = 19;
	block_index_space<4> bisc(dimensions<4>(index_range<4>(i1c, i2c)));

	mask<4> m0101, m1010, m1000, m0111, m1111;
	m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
	m1000[0] = true; m0111[1] = true; m0111[2] = true; m0111[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bisc.split(m1000, 2);
	bisc.split(m1000, 5);
	bisc.split(m1000, 7);
	bisc.split(m0111, 3);
	bisc.split(m0111, 6);
	bisc.split(m0111, 10);
	bisc.split(m0111, 13);
	bisc.split(m0111, 16);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	index<4> i0000, i0001, i0010, i0100, i1000,
		i0011, i0101, i0110, i1001, i1010, i1100,
		i0111, i1011, i1101, i1110, i1111;
	i1000[0] = 1; i0100[1] = 1; i0010[2] = 1; i0001[3] = 1;
	i0011[2] = 1; i0011[3] = 1; i0101[1] = 1; i0101[3] = 1;
	i0110[1] = 1; i0110[2] = 1; i1001[0] = 1; i1001[3] = 1;
	i1010[0] = 1; i1010[2] = 1; i1100[0] = 1; i1100[1] = 1;
	i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1011[0] = 1; i1011[2] = 1; i1011[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i1101[3] = 1;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se_part<2, double> sp1a(bisa, m11, 2), sp1b(bisb, m11, 2);
	se_part<4, double> sp2(bisc, m1111, 2);
	sp1a.add_map(i00, i11, sign);
	sp1a.add_map(i01, i10, sign);
	sp1b.add_map(i00, i11, true);
	sp1b.add_map(i01, i10, true);
	sp2.add_map(i0000, i1010, sign);
	sp2.add_map(i1010, i1111, true);
	sp2.add_map(i0101, i1111, sign);
	sp2.add_map(i0001, i1011, sign);
	sp2.add_map(i1011, i1110, true);
	sp2.add_map(i0100, i1110, sign);
	sp2.add_map(i0010, i1000, sign);
	sp2.add_map(i1000, i1101, true);
	sp2.add_map(i0111, i1101, sign);
	sp2.add_map(i0011, i1001, sign);
	sp2.add_map(i1001, i1100, true);
	sp2.add_map(i0110, i1100, sign);

	partition_set<4, double> pset1a(bisc), pset1b(bisc);
	pset1a.add_partition<2>(sp1a, permutation<2>(), m1010);
	pset1a.add_partition<2>(sp1b, permutation<2>(), m0101);
	permutation<4> perm;
	pset1b.add_partition(sp2, perm);

	symmetry_element_set<4, double> set1(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set1_ref(se_part<4, double>::k_sym_type);

	pset1a.convert(set1);
	pset1b.convert(set1_ref);
	compare_ref<4>::compare(testname, bisc, set1, set1_ref);

	mask<4> msk[1];
	msk[0][2] = true; msk[0][3] = true;
	partition_set<2, double> pset2(bisa);
	pset1a.stabilize<2, 1>(msk, pset2);

	symmetry_element_set<2, double> set2(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se_part<2, double>::k_sym_type);

	pset2.convert(set2);
	set2_ref.insert(sp1a);

	compare_ref<2>::compare(testname, bisa, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Merge two dimensions of an empty partition set
 **/
void partition_set_test::test_merge_1()
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_merge_1()";

	try {

	index<1> i1a, i2a;
	index<2> i1b, i2b;
	i2a[0] = 9;
	i2b[0] = 9; i2b[1] = 9;
	block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
	block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));

	mask<1> m1; m1[0] = true;
	bisa.split(m1, 2);
	bisa.split(m1, 5);
	bisa.split(m1, 7);

	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bisb.split(m11, 2);
	bisb.split(m11, 5);
	bisb.split(m11, 7);

	partition_set<2, double> pset1(bisb);
	partition_set<1, double> pset2(bisa);

	symmetry_element_set<1, double> set(se_part<1, double>::k_sym_type);

	pset1.merge(m11, pset2);
	pset2.convert(set);

	if (! set.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Merge two dimensions of a 2d partition set
 **/
void partition_set_test::test_merge_2(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_merge_1(bool)";

	try {

	index<1> i1a, i2a;
	index<2> i1b, i2b;
	i2a[0] = 9;
	i2b[0] = 9; i2b[1] = 9;
	block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
	block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));

	mask<1> m1; m1[0] = true;
	bisa.split(m1, 2);
	bisa.split(m1, 5);
	bisa.split(m1, 7);

	mask<2> m10, m01, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bisb.split(m11, 2);
	bisb.split(m11, 5);
	bisb.split(m11, 7);

	index<1> i0, i1;
	i1[0] = 1;
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	se_part<1, double> sp1(bisa, m1, 2);
	se_part<2, double> sp2(bisb, m11, 2);

	sp1.add_map(i0, i1, sign);
	sp2.add_map(i00, i11, sign);
	sp2.add_map(i01, i10, sign);

	partition_set<2, double> pset1(bisb);
	permutation<2> perm;
	pset1.add_partition(sp2, perm);
	partition_set<1, double> pset2(bisa);

	symmetry_element_set<1, double> set(se_part<1, double>::k_sym_type);
	symmetry_element_set<1, double> set_ref(se_part<1, double>::k_sym_type);
	set_ref.insert(sp1);

	pset1.merge(m11, pset2);
	pset2.convert(set);

	compare_ref<1>::compare(testname, bisa, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Merge two dimensions of a 3d partition set
 **/
void partition_set_test::test_merge_3(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_merge_3(bool)";

	try {

	index<2> i1a, i2a;
	index<3> i1b, i2b;
	i2a[0] = 9; i2a[1] = 15;
	i2b[0] = 9; i2b[1] = 15; i2b[2] = 9;
	block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
	block_index_space<3> bisb(dimensions<3>(index_range<3>(i1b, i2b)));

	mask<2> m01, m10, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bisa.split(m10, 2);
	bisa.split(m10, 5);
	bisa.split(m10, 7);
	bisa.split(m01, 4);
	bisa.split(m01, 8);
	bisa.split(m01, 12);

	mask<3> m101, m010, m111;
	m101[0] = true; m010[1] = true; m101[2] = true;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bisb.split(m101, 2);
	bisb.split(m101, 5);
	bisb.split(m101, 7);
	bisb.split(m010, 4);
	bisb.split(m010, 8);
	bisb.split(m010, 12);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;
	index<3> i000, i001, i010, i011, i100, i101, i110, i111;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i100[0] = 1; i011[1] = 1; i011[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<2, double> sp1(bisa, m11, 2);
	se_part<3, double> sp2(bisb, m111, 2);

	sp1.add_map(i00, i11, sign);
	sp1.add_map(i01, i10, sign);

	sp2.add_map(i000, i111, sign);
	sp2.add_map(i001, i110, sign);
	sp2.add_map(i010, i101, sign);
	sp2.add_map(i011, i100, sign);

	partition_set<3, double> pset1(bisb);
	pset1.add_partition(sp2, permutation<3>());
	partition_set<2, double> pset2(bisa);

	symmetry_element_set<2, double> set(se_part<1, double>::k_sym_type);
	symmetry_element_set<2, double> set_ref(se_part<1, double>::k_sym_type);
	set_ref.insert(sp1);

	pset1.merge(m101, pset2);
	pset2.convert(set);

	compare_ref<2>::compare(testname, bisa, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Merge three dimensions of a partition set
 **/
void partition_set_test::test_merge_4(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_merge_4(bool)";

	try {

	index<1> i1a, i2a;
	index<3> i1b, i2b;
	i2a[0] = 9;
	i2b[0] = 9; i2b[1] = 9; i2b[2] = 9;
	block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
	block_index_space<3> bisb(dimensions<3>(index_range<3>(i1b, i2b)));

	mask<1> m1; m1[0] = true;
	bisa.split(m1, 2);
	bisa.split(m1, 5);
	bisa.split(m1, 7);

	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bisb.split(m111, 2);
	bisb.split(m111, 5);
	bisb.split(m111, 7);

	index<1> i0, i1;
	i1[0] = 1;
	index<3> i000, i001, i010, i011, i100, i101, i110, i111;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i100[0] = 1; i011[1] = 1; i011[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	se_part<1, double> sp1(bisa, m1, 2);
	se_part<3, double> sp2(bisb, m111, 2);

	sp1.add_map(i0, i1, sign);
	sp2.add_map(i000, i111, sign);
	sp2.add_map(i001, i110, sign);
	sp2.add_map(i010, i101, sign);
	sp2.add_map(i011, i100, sign);

	partition_set<3, double> pset1(bisb);
	pset1.add_partition(sp2, permutation<3>());
	partition_set<1, double> pset2(bisa);

	symmetry_element_set<1, double> set(se_part<1, double>::k_sym_type);
	symmetry_element_set<1, double> set_ref(se_part<1, double>::k_sym_type);
	set_ref.insert(sp1);

	pset1.merge(m111, pset2);
	pset2.convert(set);

	compare_ref<1>::compare(testname, bisa, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
