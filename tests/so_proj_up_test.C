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


} // namespace libtensor
