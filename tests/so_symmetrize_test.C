#include <typeinfo>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_symmetrize.h>
#include <libtensor/btod/transf_double.h>
#include "so_symmetrize_test.h"
#include "compare_ref.h"

namespace libtensor {


void so_symmetrize_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


/**	\test Symmetrization of empty symmetry in 2-space.
 **/
void so_symmetrize_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_symmetrize_test::test_1()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	symmetry<2, double> sym1(bis), sym2(bis), sym2_ref(bis);

	sym2_ref.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), true));

	so_symmetrize<2, double>(sym1, permutation<2>().permute(0, 1), true).
		perform(sym2);

	compare_ref<2>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Anti-symmetrization of empty symmetry in 2-space.
 **/
void so_symmetrize_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_symmetrize_test::test_2()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	symmetry<2, double> sym1(bis), sym2(bis), sym2_ref(bis);

	sym2_ref.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), false));

	so_symmetrize<2, double>(sym1, permutation<2>().permute(0, 1), false).
		perform(sym2);

	compare_ref<2>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Symmetrization of S2*S2 to S4 in 4-space.
 **/
void so_symmetrize_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_symmetrize_test::test_3()";

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	symmetry<4, double> sym1(bis), sym2(bis), sym2_ref(bis);

	sym1.insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));
	sym1.insert(se_perm<4, double>(
		permutation<4>().permute(2, 3), true));
	sym2_ref.insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));
	sym2_ref.insert(se_perm<4, double>(permutation<4>().permute(0, 1).
		permute(1, 2).permute(2, 3), true));

	so_symmetrize<4, double>(sym1, permutation<4>().permute(1, 2), true).
		perform(sym2);

	compare_ref<4>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
