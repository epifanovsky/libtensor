#include <sstream>
#include <set>
#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/partition_set.h>
#include "compare_ref.h"
#include "partition_set_test.h"

namespace libtensor {


void partition_set_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4(true);
	test_4(false);
	test_5(true);
	test_5(false);
}

/**	\test Creation of partition sets
 **/
void partition_set_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_1()";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 19; i2[3] = 19;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m1100, m0011, m1111;
	m1100[0] = true; m1100[1] = true;
	m0011[2] = true; m0011[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1100, 2);
	bis.split(m1100, 5);
	bis.split(m1100, 7);
	bis.split(m0011, 3);
	bis.split(m0011, 6);
	bis.split(m0011, 10);
	bis.split(m0011, 13);
	bis.split(m0011, 16);

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

	se_part<4, double> sp1(bis, m1111, 2), sp2(bis, m1111, 2);
	sp1.add_map(i0000, i1111, true);
	sp1.add_map(i0101, i1010, true);
	sp1.add_map(i0110, i1001, true);
	sp2.add_map(i0001, i1000, true);
	sp2.add_map(i0010, i1101, true);
	sp2.add_map(i0100, i1011, true);
	sp2.add_map(i1000, i0111, true);
	sp2.add_map(i0011, i1100, true);

	symmetry_element_set<4, double> set1(se_part<4, double>::k_sym_type);
	symmetry_element_set<4, double> set2(se_part<4, double>::k_sym_type);
	set1.insert(sp1);
	set1.insert(sp2);

	partition_set<4, double> pset(set1);
	pset.convert(set2);

	symmetry_element_set<4, double>::iterator it = set2.begin();
	it++;
	if (it != set2.end())
		fail_test(testname, __FILE__, __LINE__, "More than one se_part.");
	
	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Adding partitions with less dimensions
 **/
void partition_set_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_2()";

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
	sp1.add_map(i0, i1, true);

	se_part<2, double> sp2(bisb, m11, 2);
	sp2.add_map(i00, i01, true);
	sp2.add_map(i01, i10, true);
	sp2.add_map(i10, i11, true);

	symmetry_element_set<2, double> set(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set_ref(se_part<2, double>::k_sym_type);
	set_ref.insert(sp2);

	permutation<1> perm;
	partition_set<2, double> pset(bisb);
	pset.add_partition(sp1, perm, m01);
	pset.add_partition(sp1, perm, m10);
	pset.convert(set);

	compare_ref<2>::compare(testname, bisb, set, set_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Intersection of two partition sets
 **/
void partition_set_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_3()";

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

	se_part<2, double> sp1(bis, m11, 2), sp2(bis, m11, 2), sp3(bis, m11, 2);
	sp1.add_map(i00, i11, true);
	sp1.add_map(i01, i10, true);
	sp2.add_map(i00, i11, true);
	sp3.add_map(i00, i11, false);

	partition_set<2, double> pset1a(bis), pset1b(bis), pset2(bis), pset3(bis);
	permutation<2> perm;
	pset1a.add_partition(sp1, perm);
	pset1b.add_partition(sp1, perm);
	pset2.add_partition(sp2, perm);
	pset3.add_partition(sp3, perm);

	symmetry_element_set<2, double> set1a(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set1b(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set1a_ref(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set1b_ref(se_part<2, double>::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se_part<2, double>::k_sym_type);
	set1a_ref.insert(sp2);
	set2_ref.insert(sp2);

	pset2.intersect(pset1a);
	pset2.convert(set2);

	compare_ref<2>::compare(testname, bis, set2, set2_ref);

	pset1a.intersect(pset2);
	pset1b.intersect(pset3);
	pset1a.convert(set1a);
	pset1b.convert(set1b);

	compare_ref<2>::compare(testname, bis, set1a, set1a_ref);
	compare_ref<2>::compare(testname, bis, set1b, set1b_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Stabilize several dimensions of a partition sets
 **/
void partition_set_test::test_4(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_4(bool)";

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

/**	\test Merge two dimensions of a partition sets
 **/
void partition_set_test::test_5(bool sign) throw(libtest::test_exception) {

	static const char *testname = "partition_set_test::test_5(bool)";

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
} // namespace libtensor
