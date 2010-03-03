#include <libtensor/symmetry/so_proj_down.h>
#include "so_proj_down_test.h"

namespace libtensor {


void so_proj_down_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
}


void so_proj_down_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_test::test_1()";

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	mask<2> msk2;
	msk2[0] = true; msk2[1] = true;
	mask<3> msk3;
	msk3[0] = true; msk3[1] = true;

	symel_cycleperm<2, double> cycle2_ref(2, msk2);
	symel_cycleperm<3, double> cycle3(2, msk3);

	mask<3> mskproj;
	mskproj[0] = true; mskproj[1] = true;
	so_projdown<3, 1, double> projdown(cycle3, mskproj, dims2);
	if(!cycle2_ref.equals(projdown.get_proj())) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect projection.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


void so_proj_down_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_test::test_2()";

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	mask<2> msk2;
	msk2[0] = true; msk2[1] = true;
	mask<3> msk3;
	msk3[0] = true; msk3[1] = true; msk3[2] = true;

	symel_cycleperm<2, double> cycle2_ref(2, msk2);
	symel_cycleperm<3, double> cycle3(3, msk3);

	mask<3> mskproj;
	mskproj[0] = true; mskproj[1] = true;
	so_projdown<3, 1, double> projdown(cycle3, mskproj, dims2);
	if(!cycle2_ref.equals(projdown.get_proj())) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect projection.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
