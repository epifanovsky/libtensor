#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_concat.h>
#include <libtensor/btod/transf_double.h>
#include "so_concat_test.h"
#include "compare_ref.h"

namespace libtensor {


void so_concat_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
}


/**	\test Invokes a concatenation of C1 in 2-space and C1 in 1-space to form a
		3-space. Expects C1 in 3-space.
 **/
void so_concat_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_concat_test::test_1()";

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));

	block_index_space<1> bis1(dims1);
	block_index_space<2> bis2(dims2);
	block_index_space<3> bis3(dims3);

	mask<1> m1; m1[0] = true;
	bis1.split(m1, 5);
	mask<2> m2; m2[0] = true; m2[1] = true;
	bis2.split(m2, 2); bis2.split(m2, 3);
	mask<3> m3a, m3b; m3a[0] = true; m3a[1] = true; m3b[2] = true;
	bis3.split(m3a, 2); bis3.split(m3a, 3); bis3.split(m3b, 5);

	symmetry<1, double> sym1(bis1);
	symmetry<2, double> sym2(bis2);
	symmetry<3, double> sym3(bis3);
	symmetry<3, double> sym3_ref(bis3);
	so_concat<2, 1, double>(sym2, sym1).perform(sym3);

	symmetry<3, double>::iterator i = sym3.begin();
	if(i != sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "i != sym3.end()");
	}

	compare_ref<3>::compare(testname, sym3, sym3_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Invokes a concatenation of two S2 in 2-space forming a 4-space.
		Expects S2*S2 in 4-space.
 **/
void so_concat_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_concat_test::test_2()";

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
	dimensions<4> dims4(index_range<4>(i4a, i4b));

	block_index_space<2> bis2(dims2);
	block_index_space<4> bis4(dims4);

	mask<2> m2; m2[0] = true; m2[1] = true;
	bis2.split(m2, 2);
	bis2.split(m2, 3);
	mask<4> m4; m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	bis4.split(m4, 2);
	bis4.split(m4, 3);

	symmetry<2, double> sym1(bis2), sym2(bis2);
	symmetry<4, double> sym3(bis4), sym3_ref(bis4);

	sym1.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), true));
	sym2.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), true));
	sym3_ref.insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));
	sym3_ref.insert(se_perm<4, double>(
		permutation<4>().permute(2, 3), true));

	so_concat<2, 2, double>(sym1, sym2).perform(sym3);

	symmetry<4, double>::iterator i = sym3.begin();
	if(i == sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "i == sym3.end()");
	}

	compare_ref<4>::compare(testname, sym3, sym3_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Invokes a concatenation of C3 in 3-space and S2 in 2-space to form
		a 5-space with a permutation. Expects C3*S2 in 5-space.
 **/
void so_concat_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_concat_test::test_3()";

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	index<3> i3a, i3b;
	i3b[0] = 8; i3b[1] = 8; i3b[2] = 8;
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	index<5> i5a, i5b;
	i5b[0] = 5; i5b[1] = 8; i5b[2] = 8; i5b[3] = 5; i5b[4] = 8;
	dimensions<5> dims5(index_range<5>(i5a, i5b));

	block_index_space<2> bis2(dims2);
	block_index_space<3> bis3(dims3);
	block_index_space<5> bis5(dims5);

	mask<2> m2; m2[0] = true; m2[1] = true;
	bis2.split(m2, 2);
	bis2.split(m2, 3);
	mask<3> m3; m3[0] = true; m3[1] = true; m3[2] = true;
 	bis3.split(m3, 3);
	bis3.split(m3, 6);
	mask<5> m5a, m5b;
	m5a[0] = true; m5a[3] = true;
	m5b[1] = true; m5b[2] = true; m5b[4] = true;
	bis5.split(m5a, 2);
	bis5.split(m5a, 3);
	bis5.split(m5b, 3);
	bis5.split(m5b, 6);

	symmetry<3, double> sym1(bis3);
	symmetry<2, double> sym2(bis2);
	symmetry<5, double> sym3(bis5);
	symmetry<5, double> sym3_ref(bis5);

	permutation<5> perm;
	perm.permute(3, 4).permute(2, 4).permute(0, 2);

	sym1.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1).permute(1, 2), true));
	sym2.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), true));
	sym3_ref.insert(se_perm<5, double>(
		permutation<5>().permute(1, 2).permute(1, 4), true));
	sym3_ref.insert(se_perm<5, double>(
		permutation<5>().permute(0, 3), true));

	so_concat<3, 2, double>(sym1, sym2, perm).perform(sym3);

	symmetry<5, double>::iterator i = sym3.begin();
	if(i == sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "i == sym3.end()");
	}

	compare_ref<5>::compare(testname, sym3, sym3_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Invokes a concatenation of vacuum with of C3 in 3-space to form a
		3-space with a permutation. Expects C3 in 3-space.
 **/
void so_concat_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_concat_test::test_4()";

	try {

	index<0> i0a, i0b;
	dimensions<0> dims0(index_range<0>(i0a, i0b));
	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 5;
	dimensions<3> dims3(index_range<3>(i3a, i3b));

	block_index_space<0> bis0(dims0);
	block_index_space<3> bis3(dims3);

	mask<3> m3; m3[0] = true; m3[1] = true; m3[2] = true;
	bis3.split(m3, 2);
	bis3.split(m3, 3);

	symmetry<0, double> sym1(bis0);
	symmetry<3, double> sym2(bis3);
	symmetry<3, double> sym3(bis3);
	symmetry<3, double> sym3_ref(bis3);

	sym2.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1).permute(1, 2), true));
	sym3_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1).permute(0, 2), true));

	permutation<3> p;
	p.permute(0, 2);
	so_concat<0, 3, double>(sym1, sym2, p).perform(sym3);

	symmetry<3, double>::iterator i = sym3.begin();
	if(i == sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "i == sym3.end()");
	}

	compare_ref<3>::compare(testname, sym3, sym3_ref);

	sym3.clear();

	so_concat<3, 0, double>(sym2, sym1, permutation<3>()).perform(sym3);

	symmetry<3, double>::iterator j = sym3.begin();
	if(j == sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "j == sym3.end()");
	}

	compare_ref<3>::compare(testname, sym3, sym3_ref);


	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
