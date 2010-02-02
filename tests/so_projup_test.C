#include <libtensor/symmetry/so_projup.h>
#include "so_projup_test.h"

namespace libtensor {


void so_projup_test::perform() throw(libtest::test_exception) {

	test_1();
}


void so_projup_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_projup_test::test_1()";

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

	symel_cycleperm<2, double> cycle2(2, msk2);
	symel_cycleperm<3, double> cycle3_ref(2, msk3);
	so_projup<2, 1, double> projup(cycle2, msk3, dims3);
	if(!cycle3_ref.equals(projup.get_proj())) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect projection.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
