#include <libtensor/symmetry/so_concat_impl_part.h>
#include <libtensor/btod/transf_double.h>
#include "so_concat_impl_part_test.h"
#include "compare_ref.h"

namespace libtensor {


void so_concat_impl_part_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2(true);
	test_2(false);
	test_3(true);
	test_3(false);
	test_4(true);
	test_4(false);
	test_5(true);
	test_5(false);
	test_6();
	test_7();
}


/**	\test Tests that a concatenation of two empty sets of partitions yields an
		empty partition set of a higher order
 **/
void so_concat_impl_part_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_part_test::test_1()";

	typedef se_part<2, double> se2_t;
	typedef se_part<3, double> se3_t;
	typedef so_concat<2, 3, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<5> i1c, i2c;
	i2c[0] = 5; i2c[1] = 5; i2c[2] = 5; i2c[3] = 5; i2c[4] = 5;
	block_index_space<5> bis5(dimensions<5>(index_range<5>(i1c, i2c)));
	mask<5> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true; m[4] = true;
	bis5.split(m, 2);
	bis5.split(m, 3);
	bis5.split(m, 5);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<5, double> set3(se2_t::k_sym_type);
	symmetry_element_set<5, double> set3_ref(se2_t::k_sym_type);

	symmetry_operation_params<so_concat_t> params(
		set1, set2, permutation<5>(), bis5, set3);

	so_concat_impl_t().perform(params);


	compare_ref<5>::compare(testname, bis5, set3, set3_ref);

	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates a set with one partition and an empty partition set
 	 	(2-space) forming a 4-space.
 **/
void so_concat_impl_part_test::test_2(bool sign)
	throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_part_test::test_2()";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<2> i1a, i2a;
	i2a[0] = 5; i2a[1] = 5;
 	block_index_space<2> bis2(dimensions<2>(index_range<2>(i1a, i2a)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis2.split(m11, 2);
	bis2.split(m11, 3);
	bis2.split(m11, 5);

	index<4> i1b, i2b;
	i2b[0] = 5; i2b[1] = 5; i2b[2] = 5; i2b[3] = 5;
 	block_index_space<4> bis4(dimensions<4>(index_range<4>(i1b, i2b)));
	mask<4> m1100, m1111;
	m1100[0] = true; m1100[1] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis4.split(m1111, 2);
	bis4.split(m1111, 3);
	bis4.split(m1111, 5);

	se2_t elem1(bis2, m11, 2);
	index<2> i00, i01, i10, i11;
	i01[1] = 1; i10[0] = 1;
	i11[1] = 1; i11[0] = 1;
	elem1.add_map(i00, i11, sign);
	elem1.add_map(i01, i10, sign);

	se4_t elem2(bis4, m1100, 2);
	index<4> i0000, i0100, i1000, i1100;
	i0100[1] = 1; i1000[0] = 1;
	i1100[1] = 1; i1100[0] = 1;
	elem2.add_map(i0000, i1100, sign);
	elem2.add_map(i0100, i1000, sign);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem1);
	set3_ref.insert(elem2);

	symmetry_operation_params<so_concat_t> params(
		set1, set2, permutation<4>(), bis4, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bis4, set3, set3_ref);


	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates a set with one partition and an empty partition set
 	 	(2-space) forming a 4-space with a permutation.
 **/
void so_concat_impl_part_test::test_3(bool sign)
	throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_part_test::test_3()";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<2> i1a, i2a;
	i2a[0] = 5; i2a[1] = 5;
 	block_index_space<2> bis2(dimensions<2>(index_range<2>(i1a, i2a)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis2.split(m11, 2);
	bis2.split(m11, 3);
	bis2.split(m11, 5);

	index<4> i1b, i2b;
	i2b[0] = 5; i2b[1] = 5; i2b[2] = 5; i2b[3] = 5;
 	block_index_space<4> bis4(dimensions<4>(index_range<4>(i1b, i2b)));
	mask<4> m1010, m1111;
	m1010[0] = true; m1010[2] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis4.split(m1111, 2);
	bis4.split(m1111, 3);
	bis4.split(m1111, 5);

	se2_t elem1(bis2, m11, 2);
	index<2> i00, i01, i10, i11;
	i01[1] = 1; i10[0] = 1;
	i11[1] = 1; i11[0] = 1;
	elem1.add_map(i00, i11, sign);
	elem1.add_map(i01, i10, sign);

	se4_t elem2(bis4, m1010, 2);
	index<4> i0000, i1000, i0010, i1010;
	i0010[2] = 1; i1000[0] = 1;
	i1010[2] = 1; i1010[0] = 1;
	elem2.add_map(i0000, i1010, sign);
	elem2.add_map(i1000, i0010, sign);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem1);
	set3_ref.insert(elem2);

	symmetry_operation_params<so_concat_t> params(
		set1, set2, permutation<4>().permute(0, 1).permute(1, 2), bis4, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bis4, set3, set3_ref);


	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates an empty partition set (2-space) and a set with one
		partition forming a 4-space with permutation.
 **/
void so_concat_impl_part_test::test_4(bool sign) throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_part_test::test_4()";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<2> i1a, i2a;
	i2a[0] = 5; i2a[1] = 5;
 	block_index_space<2> bis2(dimensions<2>(index_range<2>(i1a, i2a)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis2.split(m11, 2);
	bis2.split(m11, 3);
	bis2.split(m11, 5);

	index<4> i1b, i2b;
	i2b[0] = 5; i2b[1] = 5; i2b[2] = 5; i2b[3] = 5;
 	block_index_space<4> bis4(dimensions<4>(index_range<4>(i1b, i2b)));
	mask<4> m0110, m1111;
	m0110[1] = true; m0110[2] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis4.split(m1111, 2);
	bis4.split(m1111, 3);
	bis4.split(m1111, 5);

	se2_t elem1(bis2, m11, 2);
	index<2> i00, i01, i10, i11;
	i01[1] = 1; i10[0] = 1;
	i11[1] = 1; i11[0] = 1;
	elem1.add_map(i00, i11, sign);
	elem1.add_map(i01, i10, sign);

	se4_t elem2(bis4, m0110, 2);
	index<4> i0000, i0100, i0010, i0110;
	i0010[2] = 1; i0100[1] = 1;
	i0110[2] = 1; i0110[1] = 1;
	elem2.add_map(i0000, i0110, sign);
	elem2.add_map(i0100, i0010, sign);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set2.insert(elem1);
	set3_ref.insert(elem2);

	symmetry_operation_params< so_concat<2, 2, double> > params(
		set1, set2, permutation<4>().permute(1, 3).permute(1, 2), bis4, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bis4, set3, set3_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates two partition sets (2-space) with permutation
		[0123->3102].
 **/
void so_concat_impl_part_test::test_5(bool sign) throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_part_test::test_5()";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {


	index<2> i1a, i2a;
	i2a[0] = 5; i2a[1] = 5;
 	block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bisa.split(m11, 2);
	bisa.split(m11, 3);
	bisa.split(m11, 5);

	index<4> i1b, i2b;
	i2b[0] = 5; i2b[1] = 5; i2b[2] = 5; i2b[3] = 5;
 	block_index_space<4> bisb(dimensions<4>(index_range<4>(i1b, i2b)));
	mask<4> m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bisb.split(m1111, 2);
	bisb.split(m1111, 3);
	bisb.split(m1111, 5);

	se2_t elem1(bisa, m11, 2);
	se2_t elem2(bisa, m11, 2);
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;
	elem1.add_map(i00, i11, sign);
	elem1.add_map(i01, i10, sign);
	elem2.add_map(i00, i01, true);
	elem2.add_map(i01, i10, true);
	elem2.add_map(i10, i11, true);

	se4_t elem3(bisb, m1111, 2);
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

	elem3.add_map(i0000, i0110, sign);
	elem3.add_map(i0000, i1000, true);
	elem3.add_map(i1000, i1110, sign);
	elem3.add_map(i1000, i0001, true);
	elem3.add_map(i0001, i0111, sign);
	elem3.add_map(i0001, i1001, true);
	elem3.add_map(i1001, i1111, sign);
	elem3.add_map(i0100, i0010, sign);
	elem3.add_map(i0100, i1100, true);
	elem3.add_map(i1100, i1010, sign);
	elem3.add_map(i1100, i0101, true);
	elem3.add_map(i0101, i0011, sign);
	elem3.add_map(i0101, i1101, true);
	elem3.add_map(i1101, i1011, sign);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem1);
	set2.insert(elem2);
	set3_ref.insert(elem3);

	permutation<4> perm;
	perm.permute(2, 3).permute(0, 2);
	symmetry_operation_params<so_concat_t> params(
		set1, set2, perm, bisb, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bisb, set3, set3_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates a set with two partitions and a set with one partition
 	 	(3-space) forming a 4-space with permutation [0123->3102].
 **/
void so_concat_impl_part_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_part_test::test_6()";

	typedef se_part<1, double> se1_t;
	typedef se_part<3, double> se3_t;
	typedef se_part<4, double> se4_t;
	typedef so_concat<1, 3, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se1_t>
		so_concat_impl_t;

	try {

	index<1> i1a, i2a;
	i2a[0] = 7;
	block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
	mask<1> m1;
	m1[0] = true;
	bisa.split(m1, 2);
	bisa.split(m1, 4);
	bisa.split(m1, 6);

	index<3> i1b, i2b;
	i2b[0] = 5; i2b[1] = 5; i2b[2] = 5;
	block_index_space<3> bisb(dimensions<3>(index_range<3>(i1b, i2b)));
	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bisb.split(m111, 2);
	bisb.split(m111, 3);
	bisb.split(m111, 5);

	index<4> i1c, i2c;
	i2c[0] = 5; i2c[1] = 5; i2c[2] = 7; i2c[3] = 5;
	block_index_space<4> bisc(dimensions<4>(index_range<4>(i1c, i2c)));
	mask<4> m1101, m0010, m1111;
	m1101[0] = true; m1101[1] = true; m0010[2] = true; m1101[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bisc.split(m0010, 2);
	bisc.split(m0010, 4);
	bisc.split(m0010, 6);
	bisc.split(m1101, 2);
	bisc.split(m1101, 3);
	bisc.split(m1101, 5);

	se1_t elem1(bisa, m1, 2), elem2(bisa, m1, 4);
	index<1> i0, i1, i2, i3;
	i1[0] = 1; i2[0] = 2; i3[0] = 3;
	elem1.add_map(i0, i1, true);
	elem2.add_map(i0, i1, false);
	elem2.add_map(i0, i2, true);
	elem2.add_map(i2, i3, false);

	se3_t elem3(bisb, m111, 2);
	index<3> i000, i001, i010, i100, i011, i101, i110, i111;
	i100[0] = 1; i010[1] = 1; i001[2] = 1;
	i011[1] = 1; i011[2] = 1;
	i101[0] = 1; i101[2] = 1;
	i110[0] = 1; i110[1] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;
	elem3.add_map(i000, i111, true);
	elem3.add_map(i010, i101, true);

	se4_t elem4(bisc, m1111, 2), elem5(bisc, m0010, 4);
	index<4> i0000, i0001, i0010, i0100, i1000,
		i0011, i0101, i0110, i1001, i1010, i1100,
		i0111, i1011, i1101, i1110, i1111, i0020, i0030;
	i1000[0] = 1; i0100[1] = 1; i0010[2] = 1; i0001[3] = 1;
	i0011[2] = 1; i0011[3] = 1; i0101[1] = 1; i0101[3] = 1;
	i0110[1] = 1; i0110[2] = 1; i1001[0] = 1; i1001[3] = 1;
	i1010[0] = 1; i1010[2] = 1; i1100[0] = 1; i1100[1] = 1;
	i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1011[0] = 1; i1011[2] = 1; i1011[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i1101[3] = 1;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
	i0020[2] = 2; i0030[2] = 3;

	elem4.add_map(i0000, i0010, true);
	elem4.add_map(i0000, i1101, true);
	elem4.add_map(i0010, i1111, true);
	elem4.add_map(i0001, i0011, true);
	elem4.add_map(i0001, i1100, true);
	elem4.add_map(i0011, i1110, true);
	elem4.add_map(i0100, i0110, true);
	elem4.add_map(i0101, i0111, true);
	elem4.add_map(i1000, i1010, true);
	elem4.add_map(i1001, i1011, true);
	elem4.add_map(i1100, i1110, true);
	elem4.add_map(i1101, i1111, true);

	elem5.add_map(i0000, i0010, false);
	elem5.add_map(i0000, i0020, true);
	elem5.add_map(i0020, i0030, false);


	symmetry_element_set<1, double> set1(se1_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem1);
	set1.insert(elem2);
	set2.insert(elem3);
	set3_ref.insert(elem4);
	set3_ref.insert(elem5);

	permutation<4> perm; perm.permute(2, 3).permute(0, 2);
	symmetry_operation_params<so_concat_t> params(
		set1, set2, perm, bisc, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bisc, set3, set3_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Concatenates a set with one partition (3-space) and a set with
		two partitions forming a 4-space with permutation [0123->3102].
 **/
void so_concat_impl_part_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_part_test::test_7()";

	typedef se_part<1, double> se1_t;
	typedef se_part<3, double> se3_t;
	typedef se_part<4, double> se4_t;
	typedef so_concat<3, 1, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se3_t>
		so_concat_impl_t;

	try {

	index<1> i1a, i2a;
	i2a[0] = 7;
	block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
	mask<1> m1;
	m1[0] = true;
	bisa.split(m1, 2);
	bisa.split(m1, 4);
	bisa.split(m1, 6);

	index<3> i1b, i2b;
	i2b[0] = 5; i2b[1] = 5; i2b[2] = 5;
	block_index_space<3> bisb(dimensions<3>(index_range<3>(i1b, i2b)));
	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bisb.split(m111, 2);
	bisb.split(m111, 3);
	bisb.split(m111, 5);

	index<4> i1c, i2c;
	i2c[0] = 7; i2c[1] = 5; i2c[2] = 5; i2c[3] = 5;
	block_index_space<4> bisc(dimensions<4>(index_range<4>(i1c, i2c)));
	mask<4> m1000, m0111, m1111;
	m1000[0] = true; m0111[1] = true; m0111[2] = true; m0111[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bisc.split(m1000, 2);
	bisc.split(m1000, 4);
	bisc.split(m1000, 6);
	bisc.split(m0111, 2);
	bisc.split(m0111, 3);
	bisc.split(m0111, 5);

	se1_t elem1(bisa, m1, 2), elem2(bisa, m1, 4);
	index<1> i0, i1, i2, i3;
	i1[0] = 1; i2[0] = 2; i3[0] = 3;
	elem1.add_map(i0, i1, true);
	elem2.add_map(i0, i1, false);
	elem2.add_map(i0, i2, true);
	elem2.add_map(i2, i3, false);

	se3_t elem3(bisb, m111, 2);
	index<3> i000, i001, i010, i100, i011, i101, i110, i111;
	i100[0] = 1; i010[1] = 1; i001[2] = 1;
	i011[1] = 1; i011[2] = 1;
	i101[0] = 1; i101[2] = 1;
	i110[0] = 1; i110[1] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;
	elem3.add_map(i000, i111, true);
	elem3.add_map(i010, i101, true);

	se4_t elem4(bisc, m1111, 2), elem5(bisc, m1000, 4);
	index<4> i0000, i0001, i0010, i0100, i1000,
		i0011, i0101, i0110, i1001, i1010, i1100,
		i0111, i1011, i1101, i1110, i1111, i2000, i3000;
	i1000[0] = 1; i0100[1] = 1; i0010[2] = 1; i0001[3] = 1;
	i0011[2] = 1; i0011[3] = 1; i0101[1] = 1; i0101[3] = 1;
	i0110[1] = 1; i0110[2] = 1; i1001[0] = 1; i1001[3] = 1;
	i1010[0] = 1; i1010[2] = 1; i1100[0] = 1; i1100[1] = 1;
	i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1011[0] = 1; i1011[2] = 1; i1011[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i1101[3] = 1;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
	i2000[0] = 2; i3000[0] = 3;

	elem4.add_map(i0000, i1000, true);
	elem4.add_map(i0000, i0111, true);
	elem4.add_map(i1000, i1111, true);
	elem4.add_map(i0001, i1001, true);
	elem4.add_map(i0010, i1010, true);
	elem4.add_map(i0011, i1011, true);
	elem4.add_map(i0100, i1100, true);
	elem4.add_map(i0100, i0011, true);
	elem4.add_map(i1100, i1011, true);
	elem4.add_map(i0101, i1101, true);
	elem4.add_map(i0110, i1110, true);
	elem4.add_map(i0111, i1111, true);

	elem5.add_map(i0000, i1000, false);
	elem5.add_map(i0000, i2000, true);
	elem5.add_map(i2000, i3000, false);


	symmetry_element_set<1, double> set1(se1_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem1);
	set1.insert(elem2);
	set2.insert(elem3);
	set3_ref.insert(elem4);
	set3_ref.insert(elem5);

	permutation<4> perm; perm.permute(2, 3).permute(0, 2);
	symmetry_operation_params<so_concat_t> params(
		set2, set1, perm, bisc, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bisc, set3, set3_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}



} // namespace libtensor
