#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_proj_down.h>
#include <libtensor/btod/transf_double.h>
#include "so_proj_down_test.h"
#include "compare_ref.h"

namespace libtensor {


void so_proj_down_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


/**	\test Invokes a projection of C1 in 3-space onto 2-space.
		Expects C1 in 2-space.
 **/
void so_proj_down_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_test::test_1()";

	try {

	index<3> i1a, i1b;
	i1b[0] = 5; i1b[1] = 5; i1b[2] = 10;
	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	dimensions<3> dims1(index_range<3>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<3> bis1(dims1);
	block_index_space<2> bis2(dims2);

	symmetry<3, double> sym1(bis1);
	symmetry<2, double> sym2(bis2);
	symmetry<2, double> sym2_ref(bis2);
	mask<3> msk;
	msk[0] = true; msk[1] = true;
	so_proj_down<3, 1, double>(sym1, msk).perform(sym2);

	symmetry<2, double>::iterator i = sym2.begin();
	if(i != sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "i != sym2.end()");
	}
	compare_ref<2>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Invokes a projection of S3(+) in 3-space onto 2-space.
		Expects S2(+) in 2-space.
 **/
void so_proj_down_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_test::test_2()";

	try {

	index<3> i1a, i1b;
	i1b[0] = 5; i1b[1] = 5; i1b[2] = 10;
	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	dimensions<3> dims1(index_range<3>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<3> bis1(dims1);
	block_index_space<2> bis2(dims2);

	symmetry<3, double> sym1(bis1);
	sym1.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1).permute(1, 2), true));
	sym1.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1), true));

	symmetry<2, double> sym2(bis2);
	symmetry<2, double> sym2_ref(bis2);
	mask<3> msk;
	msk[0] = true; msk[1] = true;
	so_proj_down<3, 1, double>(sym1, msk).perform(sym2);

	sym2_ref.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), true));

	symmetry<2, double>::iterator i = sym2.begin();
	if(i == sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
	}
	compare_ref<2>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


void so_proj_down_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_test::test_3()";

	try {

	//~ index<2> i2a, i2b;
	//~ i2b[0] = 5; i2b[1] = 5;
	//~ index<3> i3a, i3b;
	//~ i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	//~ dimensions<2> dims2(index_range<2>(i2a, i2b));
	//~ dimensions<3> dims3(index_range<3>(i3a, i3b));
	//~ mask<2> msk2;
	//~ msk2[0] = true; msk2[1] = true;
	//~ mask<3> msk3;
	//~ msk3[0] = true; msk3[1] = true; msk3[2] = true;

	//~ symel_cycleperm<2, double> cycle2_ref(2, msk2);
	//~ symel_cycleperm<3, double> cycle3(3, msk3);

	//~ mask<3> mskproj;
	//~ mskproj[0] = true; mskproj[1] = true;
	//~ so_projdown<3, 1, double> projdown(cycle3, mskproj, dims2);
	//~ if(!cycle2_ref.equals(projdown.get_proj())) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Incorrect projection.");
	//~ }

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
