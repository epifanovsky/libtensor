#include <typeinfo>
#include <libtensor/symmetry/so_union.h>
#include <libtensor/btod/transf_double.h>
#include "so_union_test.h"

namespace libtensor {


void so_union_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


/**	\test Union of two empty %symmetry groups in 4-space.
 **/
void so_union_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_union_test::test_1()";

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis), sym3(bis), sym3_ref(bis);

	so_union<4, double>(sym1, sym2).perform(sym3);

	//

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test 
 **/
void so_union_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_union_test::test_2()";

	typedef symmetry_element_set<4, double> symmetry_element_set_t;

	try {

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


void so_union_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_union_test::test_3()";

	try {

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
