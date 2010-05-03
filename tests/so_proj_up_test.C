#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_proj_up.h>
#include <libtensor/btod/transf_double.h>
#include "so_proj_up_test.h"
#include "compare_ref.h"

namespace libtensor {


void so_proj_up_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
}


/**	\test Invokes a projection of C1 in 2-space onto 3-space.
		Expects C1 in 3-space.
 **/
void so_proj_up_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_test::test_1()";

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));

	block_index_space<2> bis2(dims2);
	block_index_space<3> bis3(dims3);

	mask<2> m2;
	m2[0] = true; m2[1] = true;
	bis2.split(m2, 2);
	bis2.split(m2, 3);
	mask<3> m3a, m3b;
	m3a[0] = true; m3a[1] = true; m3b[2] = true;
	bis3.split(m3a, 2);
	bis3.split(m3a, 3);
	bis3.split(m3b, 5);

	symmetry<2, double> sym1(bis2);
	symmetry<3, double> sym2(bis3);
	symmetry<3, double> sym2_ref(bis3);
	mask<3> msk;
	msk[0] = true; msk[1] = true;
	so_proj_up<2, 1, double>(sym1, msk).perform(sym2);

	symmetry<3, double>::iterator i = sym2.begin();
	if(i != sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "i != sym2.end()");
	}

	compare_ref<3>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Invokes a projection of S2 in 2-space onto 3-space.
		Expects S2 in 3-space.
 **/
void so_proj_up_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_test::test_2()";

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));

	block_index_space<2> bis2(dims2);
	block_index_space<3> bis3(dims3);

	mask<2> m2;
	m2[0] = true; m2[1] = true;
	bis2.split(m2, 2);
	bis2.split(m2, 3);
	mask<3> m3a, m3b;
	m3a[0] = true; m3a[1] = true; m3b[2] = true;
	bis3.split(m3a, 2);
	bis3.split(m3a, 3);
	bis3.split(m3b, 5);

	symmetry<2, double> sym1(bis2);
	symmetry<3, double> sym2(bis3);
	symmetry<3, double> sym2_ref(bis3);

	sym1.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), true));
	sym2_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1), true));

	mask<3> msk;
	msk[0] = true; msk[1] = true;
	so_proj_up<2, 1, double>(sym1, msk).perform(sym2);

	symmetry<3, double>::iterator i = sym2.begin();
	if(i == sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
	}

	compare_ref<3>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Invokes a projection of S2 in 2-space onto 3-space with
		a permutation. Expects S2 in 3-space.
 **/
void so_proj_up_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_test::test_3()";

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));

	block_index_space<2> bis2(dims2);
	block_index_space<3> bis3(dims3);

	mask<2> m2;
	m2[0] = true; m2[1] = true;
	bis2.split(m2, 2);
	bis2.split(m2, 3);
	mask<3> m3a, m3b;
	m3a[0] = true; m3a[1] = true; m3b[2] = true;
	bis3.split(m3a, 2);
	bis3.split(m3a, 3);
	bis3.split(m3b, 5);

	symmetry<2, double> sym1(bis2);
	symmetry<3, double> sym2(bis3);
	symmetry<3, double> sym2_ref(bis3);

	sym1.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), true));
	sym2_ref.insert(se_perm<3, double>(
		permutation<3>().permute(1, 2), true));

	mask<3> msk;
	msk[1] = true; msk[2] = true;
	so_proj_up<2, 1, double>(sym1, permutation<2>(), msk).perform(sym2);

	symmetry<3, double>::iterator i = sym2.begin();
	if(i == sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
	}

	compare_ref<3>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Invokes a projection of S2*S2 in 4-space onto 6-space with
		a permutation. Expects S2*S2 in 6-space.
 **/
void so_proj_up_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_test::test_4()";

	try {

	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 6; i4b[2] = 5; i4b[3] = 6;
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	index<6> i6a, i6b;
	i6b[0] = 5; i6b[1] = 5; i6b[2] = 10; i6b[3] = 6; i6b[4] = 6; i6b[5] = 9;
	dimensions<6> dims6(index_range<6>(i6a, i6b));

	block_index_space<4> bis4(dims4);
	block_index_space<6> bis6(dims6);

	mask<4> m4a, m4b;
	m4a[0] = true; m4b[1] = true; m4a[2] = true; m4b[3] = true;
	bis4.split(m4a, 2);
	bis4.split(m4a, 3);
	bis4.split(m4b, 3);
	mask<6> m6a, m6b, m6c, m6d;
	m6a[0] = true; m6a[1] = true; m6c[2] = true;
	m6b[3] = true; m6b[4] = true; m6d[5] = true;
	bis6.split(m6a, 2);
	bis6.split(m6a, 3);
	bis6.split(m6b, 3);
	bis6.split(m6c, 4);
	bis6.split(m6c, 8);
	bis6.split(m6d, 2);
	bis6.split(m6d, 7);

	symmetry<4, double> sym1(bis4);
	symmetry<6, double> sym2(bis6);
	symmetry<6, double> sym2_ref(bis6);

	sym1.insert(se_perm<4, double>(
		permutation<4>().permute(0, 2), true));
	sym1.insert(se_perm<4, double>(
		permutation<4>().permute(1, 3), true));
	sym2_ref.insert(se_perm<6, double>(
		permutation<6>().permute(0, 1), true));
	sym2_ref.insert(se_perm<6, double>(
		permutation<6>().permute(3, 4), true));

	mask<6> m;
	m[0] = true; m[1] = true; m[3] = true; m[4] = true;
	so_proj_up<4, 2, double>(sym1, permutation<4>().permute(1, 2), m).
		perform(sym2);

	symmetry<6, double>::iterator i = sym2.begin();
	if(i == sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
	}

	compare_ref<6>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
