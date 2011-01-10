#include <sstream>
#include <string>
#include <libtensor/core/index.h>
#include "index_test.h"

namespace libtensor {

void index_test::perform() throw(libtest::test_exception) {
	test_ctor();
	test_less();
	test_print();
	test_op();
}

void index_test::test_ctor() throw(libtest::test_exception) {
	index<2> i1;
}

void index_test::test_less() throw(libtest::test_exception) {
	index<2> i1, i2;

	i1[0] = 1; i1[1] = 1;
	i2[0] = 2; i2[1] = 2;
	if(!i1.less(i2)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less doesn't return (1,1)<(2,2)");
	}
	if(i2.less(i1)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less returns (2,2)<(1,1)");
	}
	i1[0] = 2;
	if(!i1.less(i2)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less doesn't return (2,1)<(2,2)");
	}
	if(i2.less(i1)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less returns (2,2)<(2,1)");
	}
	i1[1] = 2;
	if(i1.less(i2)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less returns (2,2)<(2,2)");
	}

	i1[0] = 0; i1[1] = 10;
	i2[0] = 10; i2[1] = 12;
	if(!i1.less(i2)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less returns (10,12)<(0,10)");
	}
	i1[1] = 12;
	if(!i1.less(i2)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less returns (10,12)<(0,12)");
	}

}

void index_test::test_print() throw(libtest::test_exception) {
	std::ostringstream ss1;
	index<1> i1;
	ss1 << i1;
	if(ss1.str().compare("[0]")!=0) {
		std::ostringstream err;
		err << "output error: expected \'[0]\', received \'";
		err << ss1.str() << "\'";
		fail_test("index_test::test_print()", __FILE__, __LINE__,
			err.str().c_str());
	}

	std::ostringstream ss2;
	i1[0]=25;
	ss2 << i1;
	if(ss2.str().compare("[25]")!=0) {
		std::ostringstream err;
		err << "output error: expected \'[25]\', received \'";
		err << ss2.str() << "\'";
		fail_test("index_test::test_print()", __FILE__, __LINE__,
			err.str().c_str());
	}

	std::ostringstream ss3;
	index<1> i1a; i1a[0]=3;
	ss3 << i1a << i1;
	if(ss3.str().compare("[3][25]")!=0) {
		std::ostringstream err;
		err << "output error: expected \'[3][25]\', received \'";
		err << ss3.str() << "\'";
		fail_test("index_test::test_print()", __FILE__, __LINE__,
			err.str().c_str());
	}

	std::ostringstream ss4;
	index<2> i2;
	ss4 << i2;
	if(ss4.str().compare("[0,0]")!=0) {
		std::ostringstream err;
		err << "output error: expected \'[0,0]\', received \'";
		err << ss4.str() << "\'";
		fail_test("index_test::test_print()", __FILE__, __LINE__,
			err.str().c_str());
	}

	std::ostringstream ss5;
	i2[0]=3; i2[1]=4;
	ss5 << i2;
	if(ss5.str().compare("[3,4]")!=0) {
		std::ostringstream err;
		err << "output error: expected \'[3,4]\', received \'";
		err << ss5.str() << "\'";
		fail_test("index_test::test_print()", __FILE__, __LINE__,
			err.str().c_str());
	}

}

void index_test::test_op() throw(libtest::test_exception) {

	index<2> i1, i2, i3, i4;
	i1[0] = 3; i1[1] = 5;
	i2[0] = 3; i2[1] = 5;
	i3[0] = 0; i3[1] = 0;
	i4[0] = 3; i4[1] = 6;

	if (! (i1 == i2))
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator==(i1, i2)");

	if (i1 != i2)
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator!=(i1, i2)");

	if (i1 == i3)
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator==(i1, i3)");

	if (! (i1 != i3))
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator!=(i1, i3)");

	if (i1 < i2)
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator<(i1, i2)");

	if (i1 < i3)
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator<(i1, i3)");

	if (! (i1 < i4))
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator<(i1, i4)");

	if (! (i1 <= i2))
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator<=(i1, i2)");

	if (i1 <= i3)
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator<=(i1, i3)");

	if (! (i1 <= i4))
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator<=(i1, i4)");

	if (i1 > i2)
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator>(i1, i2)");

	if (! (i1 > i3))
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator>(i1, i3)");

	if (i1 > i4)
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator>(i1, i4)");

	if (! (i1 >= i2))
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator>=(i1, i2)");

	if (! (i1 >= i3))
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator>=(i1, i3)");

	if (i1 > i4)
		fail_test("index_test::test_op()",
				__FILE__, __LINE__, "operator>=(i1, i4)");
}

} // namespace libtensor

