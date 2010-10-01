#include <libtensor/symmetry/so_stabilize_impl_part.h>
#include <libtensor/btod/transf_double.h>
#include "compare_ref.h"
#include "so_stabilize_impl_part_test.h"


namespace libtensor {

void so_stabilize_impl_part_test::perform() throw(libtest::test_exception) {

	test_1a();
	test_1b();
	test_2(true);
	test_2(false);
	test_3();
	test_4();
	test_5a(true);
	test_5a(false);
	test_5b(true);
	test_5b(false);
	test_5c(true);
	test_5c(false);
	test_5d(true);
	test_5d(false);

}


/**	\test Tests that a projection of an empty group yields an empty group
		of a lower order
 **/
void so_stabilize_impl_part_test::test_1a() throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_part_test::test_1a()";

	typedef se_part<4, double> se4_t;
	typedef se_part<2, double> se2_t;
	typedef so_stabilize<4, 2, 1, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	try {

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	mask<4> msk[1]; msk[0][2] = true; msk[0][3] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Tests that a double projection of an empty group yields an empty group
		of a lower order
 **/
void so_stabilize_impl_part_test::test_1b() throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_part_test::test_1b()";

	typedef se_part<4, double> se4_t;
	typedef se_part<2, double> se2_t;
	typedef so_stabilize<4, 2, 2, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	try {

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	mask<4> msk[2]; msk[1][2] = true; msk[0][3] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Projection of a 2-space on a 1-space.
 **/
void so_stabilize_impl_part_test::test_2(bool sign)
	throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_part_test::test_2(bool)";

	typedef se_part<2, double> se2_t;
	typedef se_part<1, double> se1_t;
	typedef so_stabilize<2, 1, 1, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se2_t>
		so_stabilize_impl_t;

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

	mask<2> m[1];
	m[0][1] = true;
	symmetry_operation_params<so_stabilize_t> params(set2, m, set1);
	so_stabilize_impl_t().perform(params);

	if(set1.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<1>::compare(testname, bis1, set1, set1_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projection of a 4-space onto a 2-space.
 **/
void so_stabilize_impl_part_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_part_test::test_3()";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_stabilize<4, 2, 1, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 9; i2b[1] = 9;
	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	mask<2> m11; m11[0] = true; m11[1] = true;
	bis2.split(m11, 2);
	bis2.split(m11, 3);
	bis2.split(m11, 5);
	bis2.split(m11, 7);
	bis2.split(m11, 8);

	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 5; i4b[2] = 9; i4b[3] = 9;
 	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));
	mask<4> m0011, m1100, m1111;
	m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis4.split(m0011, 2);
	bis4.split(m0011, 3);
	bis4.split(m0011, 5);
	bis4.split(m0011, 7);
	bis4.split(m0011, 8);
	bis4.split(m1100, 2);
	bis4.split(m1100, 3);
	bis4.split(m1100, 5);

	se4_t elem1(bis4, m1111, 2);
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
	elem1.add_map(i0000, i1111, true);
	elem1.add_map(i0011, i1100, true);
	elem1.add_map(i0001, i1110, true);
	elem1.add_map(i0010, i1101, true);

	se2_t elem2(bis2, m11, 2);
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;
	elem2.add_map(i00, i11, true);
	elem2.add_map(i01, i10, true);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se2_t::k_sym_type);

	set1.insert(elem1);
	set2_ref.insert(elem2);

	mask<4> msk[1];
	msk[0] = m1100;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<2>::compare(testname, bis2, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Projection of a 4-space onto a 2-space in two steps.
 **/
void so_stabilize_impl_part_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_part_test::test_4()";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_stabilize<4, 2, 2, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 9;
	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	mask<2> m01, m10, m11;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bis2.split(m10, 2);
	bis2.split(m10, 3);
	bis2.split(m10, 5);
	bis2.split(m01, 2);
	bis2.split(m01, 3);
	bis2.split(m01, 5);
	bis2.split(m01, 7);
	bis2.split(m01, 8);

	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 5; i4b[2] = 9; i4b[3] = 9;
 	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));
	mask<4> m0011, m1100, m1111;
	m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis4.split(m0011, 2);
	bis4.split(m0011, 3);
	bis4.split(m0011, 5);
	bis4.split(m0011, 7);
	bis4.split(m0011, 8);
	bis4.split(m1100, 2);
	bis4.split(m1100, 3);
	bis4.split(m1100, 5);

	se4_t elem1(bis4, m1111, 2);
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

	elem1.add_map(i0000, i1010, true);
	elem1.add_map(i0001, i1011, true);
	elem1.add_map(i0100, i1110, true);
	elem1.add_map(i0101, i1111, true);
	elem1.add_map(i0010, i1000, true);
	elem1.add_map(i0111, i1101, true);

	se2_t elem2(bis2, m11, 2);
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;
	elem2.add_map(i00, i11, true);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se2_t::k_sym_type);

	set1.insert(elem1);
	set2_ref.insert(elem2);

	mask<4> msk[2];
	msk[0][3] = true;
	msk[1][1] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<2>::compare(testname, bis2, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Projection of a 4-space onto a 2-space in one step with partial
		partitioning (only not projected dims are partitioned)

 **/
void so_stabilize_impl_part_test::test_5a(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_part_test::test_5a(bool)";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_stabilize<4, 2, 1, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis2.split(m11, 2);
	bis2.split(m11, 3);
	bis2.split(m11, 5);

	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 5; i4b[2] = 9; i4b[3] = 9;
 	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));
	mask<4> m0011, m1100;
	m1100[0] = true; m1100[1] = true;
	m0011[2] = true; m0011[3] = true;
	bis4.split(m0011, 2);
	bis4.split(m0011, 3);
	bis4.split(m0011, 5);
	bis4.split(m0011, 7);
	bis4.split(m0011, 8);
	bis4.split(m1100, 2);
	bis4.split(m1100, 3);
	bis4.split(m1100, 5);

	se4_t elem1(bis4, m1100, 2);
	index<4> i0000, i0100, i1000, i1100;
	i1000[0] = 1; i0100[1] = 1;
	i1100[0] = 1; i1100[1] = 1;

	elem1.add_map(i0000, i1100, sign);
	elem1.add_map(i0100, i1000, sign);

	se2_t elem2(bis2, m11, 2);
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

	mask<4> msk[1];
	msk[0][2] = true; msk[0][3] = true;

	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<2>::compare(testname, bis2, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Projection of a 4-space onto a 2-space in one step with partial
		partitioning (only one partitioned dim, not projected)
 **/
void so_stabilize_impl_part_test::test_5b(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_part_test::test_5b(bool)";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_stabilize<4, 2, 1, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 9;
	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	mask<2> m01, m10;
	m10[0] = true; m01[1] = true;
	bis2.split(m10, 2);
	bis2.split(m10, 3);
	bis2.split(m10, 5);
	bis2.split(m01, 2);
	bis2.split(m01, 3);
	bis2.split(m01, 5);
	bis2.split(m01, 7);
	bis2.split(m01, 8);

	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 9; i4b[2] = 9; i4b[3] = 9;
 	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));
	mask<4> m0111, m1000;
	m1000[0] = true; m0111[1] = true; m0111[2] = true; m0111[3] = true;
	bis4.split(m0111, 2);
	bis4.split(m0111, 3);
	bis4.split(m0111, 5);
	bis4.split(m0111, 7);
	bis4.split(m0111, 8);
	bis4.split(m1000, 2);
	bis4.split(m1000, 3);
	bis4.split(m1000, 5);

	se4_t elem1(bis4, m1000, 2);
	index<4> i0000, i1000;
	i1000[0] = 1;

	elem1.add_map(i0000, i1000, sign);

	se2_t elem2(bis2, m10, 2);
	index<2> i00, i10;
	i10[0] = 1;
	elem2.add_map(i00, i10, sign);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2_ref(se2_t::k_sym_type);

	set1.insert(elem1);
	set2_ref.insert(elem2);

	mask<4> msk[1];
	msk[0][2] = true; msk[0][3] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<2>::compare(testname, bis2, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Projection of a 4-space onto a 2-space in one step with partial
		partitioning (only one partitioned dim, projected)
 **/
void so_stabilize_impl_part_test::test_5c(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_part_test::test_5c(bool)";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_stabilize<4, 2, 1, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis2.split(m11, 2);
	bis2.split(m11, 3);
	bis2.split(m11, 5);

	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 5; i4b[2] = 9; i4b[3] = 9;
 	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));
	mask<4> m0011, m1100, m0001;
	m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
	m0001[3] = true;
	bis4.split(m0011, 2);
	bis4.split(m0011, 3);
	bis4.split(m0011, 5);
	bis4.split(m0011, 7);
	bis4.split(m0011, 8);
	bis4.split(m1100, 2);
	bis4.split(m1100, 3);
	bis4.split(m1100, 5);

	se4_t elem1(bis4, m0001, 2);
	index<4> i0000, i0001; i0001[3] = 1;
	elem1.add_map(i0000, i0001, sign);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	set1.insert(elem1);

	mask<4> msk[1];
	msk[0][2] = true; msk[0][3] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	bool found = false;
	try {

	so_stabilize_impl_t().perform(params);

	} catch(exception &e) {
		found = true;
	}

	if (! found) {
		fail_test(testname, __FILE__, __LINE__, "No exception.");
	}
}

/**	\test Projection of a 4-space onto a 2-space in one step with partial
		partitioning (only projected dims are partitioned)
 **/
void so_stabilize_impl_part_test::test_5d(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_part_test::test_5d(bool)";

	typedef se_part<2, double> se2_t;
	typedef se_part<4, double> se4_t;
	typedef so_stabilize<4, 2, 1, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis2.split(m11, 2);
	bis2.split(m11, 3);
	bis2.split(m11, 5);

	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 5; i4b[2] = 9; i4b[3] = 9;
 	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));
	mask<4> m0011, m1100;
	m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
	bis4.split(m0011, 2);
	bis4.split(m0011, 3);
	bis4.split(m0011, 5);
	bis4.split(m0011, 7);
	bis4.split(m0011, 8);
	bis4.split(m1100, 2);
	bis4.split(m1100, 3);
	bis4.split(m1100, 5);

	se4_t elem1(bis4, m0011, 2);
	index<4> i0000, i0001, i0010, i0011;
	i0010[2] = 1; i0001[3] = 1;
	i0011[2] = 1; i0011[3] = 1;

	elem1.add_map(i0000, i0011, sign);
	elem1.add_map(i0001, i0010, sign);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	set1.insert(elem1);

	mask<4> msk[1];
	msk[0][2] = true; msk[0][3] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(! set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
