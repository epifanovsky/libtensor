#include <libtensor/symmetry/so_merge_impl_part.h>
#include <libtensor/btod/transf_double.h>
#include "compare_ref.h"
#include "so_merge_impl_part_test.h"


namespace libtensor {

void so_merge_impl_part_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2(true);
	test_2(false);
	test_3(true);
	test_3(false);
	test_4(true);
	test_4(false);
}


/**	\test Tests that a merge of 2 dim of an empty partition set yields an
		empty partition set of lower order
 **/
void so_merge_impl_part_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_part_test::test_1()";

	typedef se_part<4, double> se4_t;
	typedef se_part<3, double> se3_t;
	typedef so_merge<4, 2, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se4_t>
		so_merge_impl_t;

	try {

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);

	mask<4> msk; msk[2] = true; msk[3] = true;
	symmetry_operation_params<so_merge_t> params(set1, msk, set2);

	so_merge_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Merge of 2 dim of a 2-space on a 1-space.
 **/
void so_merge_impl_part_test::test_2(bool sign)
	throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_part_test::test_2(bool)";

	typedef se_part<2, double> se2_t;
	typedef se_part<1, double> se1_t;
	typedef so_merge<2, 2, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se2_t>
		so_merge_impl_t;

	try {

	index<1> i1a, i1b;
	i1b[0] = 5;
	block_index_space<1> bis1(dimensions<1>(index_range<1>(i1a, i1b)));
	mask<1> m1; m1[0] = true;
	bis1.split(m1, 2);
	bis1.split(m1, 3);
	bis1.split(m1, 5);

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis2.split(m11, 2);
	bis2.split(m11, 3);
	bis2.split(m11, 5);

	se1_t elem1(bis1, m1, 2);
	index<1> i0, i1;
	i1[0] = 1;
	elem1.add_map(i0, i1, sign);

	se2_t elem2(bis2, m11, 2);
	index<2> i00, i01, i10, i11;
	i01[1] = 1; i10[0] = 1;
	i11[1] = 1; i11[0] = 1;
	elem2.add_map(i00, i11, sign);
	elem2.add_map(i01, i10, sign);

	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<1, double> set1(se1_t::k_sym_type);
	symmetry_element_set<1, double> set1_ref(se1_t::k_sym_type);
	set1_ref.insert(elem1);
	set2.insert(elem2);

	symmetry_operation_params<so_merge_t> params(set2, m11, set1);
	so_merge_impl_t().perform(params);

	if(set1.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<1>::compare(testname, bis1, set1, set1_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Merge of 2 dim of a 4-space onto a 3-space.
 **/
void so_merge_impl_part_test::test_3(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_part_test::test_3()";

	typedef se_part<3, double> se3_t;
	typedef se_part<4, double> se4_t;
	typedef so_merge<4, 2, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se4_t>
		so_merge_impl_t;

	try {

	index<3> i2a, i2b;
	i2b[0] = 5; i2b[1] = 9; i2b[2] = 9;
	block_index_space<3> bisa(dimensions<3>(index_range<3>(i2a, i2b)));
	mask<3> m011, m100, m111;
	m100[0] = true; m011[1] = true; m011[2] = true;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bisa.split(m011, 2);
	bisa.split(m011, 3);
	bisa.split(m011, 5);
	bisa.split(m011, 7);
	bisa.split(m011, 8);
	bisa.split(m100, 2);
	bisa.split(m100, 3);
	bisa.split(m100, 5);

	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 5; i4b[2] = 9; i4b[3] = 9;
 	block_index_space<4> bisb(dimensions<4>(index_range<4>(i4a, i4b)));
	mask<4> m0011, m1100, m1111;
	m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bisb.split(m0011, 2);
	bisb.split(m0011, 3);
	bisb.split(m0011, 5);
	bisb.split(m0011, 7);
	bisb.split(m0011, 8);
	bisb.split(m1100, 2);
	bisb.split(m1100, 3);
	bisb.split(m1100, 5);

	se4_t elem1(bisb, m1111, 2);
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
	elem1.add_map(i0000, i1111, sign);
	elem1.add_map(i0011, i1100, sign);
	elem1.add_map(i0001, i1110, sign);
	elem1.add_map(i0010, i1101, sign);

	se3_t elem2(bisa, m111, 2);
	index<3> i000, i001, i010, i100, i011, i101, i110, i111;
	i100[0] = 1; i011[1] = 1; i011[2] = 1;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i110[0] = 1; i110[1] = 1; i001[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;
	elem2.add_map(i000, i111, sign);
	elem2.add_map(i001, i110, sign);
	elem2.add_map(i010, i101, sign);
	elem2.add_map(i011, i100, sign);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se3_t::k_sym_type);

	set1.insert(elem1);
	set2_ref.insert(elem2);

	symmetry_operation_params<so_merge_t> params(set1, m1100, set2);

	so_merge_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<3>::compare(testname, bisa, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Merge of 3 dim of a 4-space onto a 2-space.
 **/
void so_merge_impl_part_test::test_4(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_part_test::test_3()";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_merge<4, 3, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se4_t>
		so_merge_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 9;
	block_index_space<2> bisa(dimensions<2>(index_range<2>(i2a, i2b)));
	mask<2> m01, m10, m11;
	m10[0] = true; m01[1] = true; m01[2] = true;
	m11[0] = true; m11[1] = true; m11[2] = true;
	bisa.split(m01, 2);
	bisa.split(m01, 3);
	bisa.split(m01, 5);
	bisa.split(m01, 7);
	bisa.split(m01, 8);
	bisa.split(m10, 2);
	bisa.split(m10, 3);
	bisa.split(m10, 5);

	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 5; i4b[2] = 9; i4b[3] = 5;
 	block_index_space<4> bisb(dimensions<4>(index_range<4>(i4a, i4b)));
	mask<4> m0010, m1101, m1111;
	m1101[0] = true; m1101[1] = true; m0010[2] = true; m1101[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bisb.split(m0010, 2);
	bisb.split(m0010, 3);
	bisb.split(m0010, 5);
	bisb.split(m0010, 7);
	bisb.split(m0010, 8);
	bisb.split(m1101, 2);
	bisb.split(m1101, 3);
	bisb.split(m1101, 5);

	se4_t elem1(bisb, m1111, 2);
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
	elem1.add_map(i0000, i1111, sign);
	elem1.add_map(i0011, i1100, sign);
	elem1.add_map(i0001, i1110, sign);
	elem1.add_map(i0010, i1101, sign);

	se2_t elem2(bisa, m11, 2);
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;
	elem2.add_map(i00, i11, sign);
	elem2.add_map(i01, i10, sign);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se2_t::k_sym_type);

	set1.insert(elem1);
	set2_ref.insert(elem2);

	symmetry_operation_params<so_merge_t> params(set1, m1101, set2);

	so_merge_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<2>::compare(testname, bisa, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
