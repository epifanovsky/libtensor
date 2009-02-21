#include "dimensions_test.h"

namespace libtensor {

void dimensions_test::perform() throw(libtest::test_exception) {
	test_ctor();
	test_inc_index();
}

void dimensions_test::test_ctor() throw(libtest::test_exception) {
	index i1a(2), i1b(2);
	i1b[0] = 1; i1b[1] = 2;
	index_range ir1(i1a, i1b); // Indexes run from (0,0) to (1,2)
	dimensions d1(ir1);

	if(d1.get_order() != 2) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect number of dimensions in d1");
	}
	if(d1[0] != 2) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect number of elements along d1[0]");
	}
	if(d1[1] != 3) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect number of elements along d1[1]");
	}
	if(d1.get_size() != 6) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect total number of elements in d1");
	}

	dimensions d2(d1);

	if(d2.get_order() != 2) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect number of dimensions in d2");
	}
	if(d2[0] != 2) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect number of elements along d2[0]");
	}
	if(d2[1] != 3) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect number of elements along d2[1]");
	}
	if(d2.get_size() != 6) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect total number of elements in d2");
	}

}

void dimensions_test::test_inc_index() throw(libtest::test_exception) {
	index i1(4), i2(4);
	i2[0]=1; i2[1]=1; i2[2]=1; i2[3]=1;
	index_range ir(i1,i2);
	dimensions d(ir);

	if(!d.inc_index(i1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0,0,0,0) doesn't return true");
	}
	if(!(i1[0]==0 && i1[1]==0 && i1[2]==0 && i1[3]==1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0,0,0,0) doesn't return (0,0,0,1)");
	}

	i1[0]=1; i1[1]=1; i1[2]=0; i1[3]=0;
	if(!d.inc_index(i1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(1,1,0,0) doesn't return true");
	}
	if(!(i1[0]==1 && i1[1]==1 && i1[2]==0 && i1[3]==1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0,0,0,0) doesn't return (1,1,0,1)");
	}
	if(!d.inc_index(i1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(1,1,0,1) doesn't return true");
	}
	if(!(i1[0]==1 && i1[1]==1 && i1[2]==1 && i1[3]==0)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0,0,0,0) doesn't return (1,1,1,0)");
	}
	i1[3]=1;
	if(d.inc_index(i1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(1,1,1,1) doesn't return false");
	}
	if(!(i1[0]==1 && i1[1]==1 && i1[2]==1 && i1[3]==1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(1,1,1,1) doesn't preserve the index");
	}
	i1[0]=2; i1[1]=2; i1[2]=2; i1[3]=2;
	if(d.inc_index(i1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(2,2,2,2) doesn't return false");
	}
	if(!(i1[0]==2 && i1[1]==2 && i1[2]==2 && i1[3]==2)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(2,2,2,2) doesn't preserve the index");
	}
}

} // namespace libtensor

