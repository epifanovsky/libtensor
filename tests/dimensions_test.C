#include <libtensor.h>
#include "dimensions_test.h"

namespace libtensor {

void dimensions_test::perform() throw(libtest::test_exception) {
	test_ctor();
	test_contains();
	test_inc_index();
	test_abs_index();
}

void dimensions_test::test_ctor() throw(libtest::test_exception) {
	index<2> i1a, i1b;
	i1b[0] = 1; i1b[1] = 2;
	index_range<2> ir1(i1a, i1b); // Indexes run from (0,0) to (1,2)
	dimensions<2> d1(ir1);

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

	dimensions<2> d2(d1);

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

void dimensions_test::test_contains() throw(libtest::test_exception) {
	index<2> i1, i2; i2[0]=10; i2[1]=12;
	dimensions<2> d1(index_range<2>(i1, i2));

	i1[0] = 0; i1[1] = 0;
	if(!d1.contains(i1)) {
		fail_test("dimensions_test::test_contains()", __FILE__,
			__LINE__, "(11,13).contains(0,0) returns false");
	}

	i1[1] = 1;
	if(!d1.contains(i1)) {
		fail_test("dimensions_test::test_contains()", __FILE__,
			__LINE__, "(11,13).contains(0,1) returns false");
	}

	i1[1] = 12;
	if(!d1.contains(i1)) {
		fail_test("dimensions_test::test_contains()", __FILE__,
			__LINE__, "(11,13).contains(0,12) returns false");
	}

	i1[1] = 13;
	if(d1.contains(i1)) {
		fail_test("dimensions_test::test_contains()", __FILE__,
			__LINE__, "(11,13).contains(0,13) returns true");
	}

	i1[0] = 1; i1[1] = 0;
	if(!d1.contains(i1)) {
		fail_test("dimensions_test::test_contains()", __FILE__,
			__LINE__, "(11,13).contains(1,0) returns false");
	}

	i1[0] = 10;
	if(!d1.contains(i1)) {
		fail_test("dimensions_test::test_contains()", __FILE__,
			__LINE__, "(11,13).contains(10,0) returns false");
	}

	i1[0] = 11;
	if(d1.contains(i1)) {
		fail_test("dimensions_test::test_contains()", __FILE__,
			__LINE__, "(11,13).contains(11,0) returns true");
	}

	i1[1] = 100;
	if(d1.contains(i1)) {
		fail_test("dimensions_test::test_contains()", __FILE__,
			__LINE__, "(11,13).contains(11,100) returns true");
	}
}

void dimensions_test::test_inc_index() throw(libtest::test_exception) {
	index<4> i1, i2;
	i2[0]=1; i2[1]=1; i2[2]=1; i2[3]=1;
	index_range<4> ir(i1,i2);
	dimensions<4> d(ir);

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

	index<2> j1, j2; j2[0]=10; j2[1]=12;
	dimensions<2> d2(index_range<2>(j1, j2));

	j1[0]=0; j1[1]=11;
	if(!d2.inc_index(j1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0,11) doesn't return true");
	}
	if(!(j1[0]==0 && j1[1]==12)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0,11) doesn't return (0,12)");
	}
	if(!d2.inc_index(j1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0,12) doesn't return true");
	}
	if(!(j1[0]==1 && j1[1]==0)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0,12) doesn't return (1,0)");
	}

	index<1> k1, k2; k2[0]=5;
	dimensions<1> d3(index_range<1>(k1, k2));

	if(!d3.inc_index(k1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0) doesn't return true");
	}
	if(k1[0]!=1) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(0) doesn't return (1)");
	}
	k1[0]=4;
	if(!d3.inc_index(k1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(4) doesn't return true");
	}
	if(k1[0]!=5) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(4) doesn't return (5)");
	}
	if(d3.inc_index(k1)) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(5) doesn't return false");
	}
	if(k1[0]!=5) {
		fail_test("dimensions_test::test_inc_index()", __FILE__,
			__LINE__, "inc(4) doesn't preserve the index");
	}


}

void dimensions_test::test_abs_index() throw(libtest::test_exception) {
	index<2> i1, i2;
	i2[0]=9; i2[1]=9;
	index_range<2> ir(i1,i2);
	dimensions<2> d(ir);

	if(d.abs_index(i1)!=0) {
		fail_test("dimensions::test_abs_index()", __FILE__, __LINE__,
			"abs(0,0) in (10,10) doesn't return 0");
	}
	i1[0]=1; i1[1]=0;
	if(d.abs_index(i1)!=10) {
		fail_test("dimensions::test_abs_index()", __FILE__, __LINE__,
			"abs(1,0) in (10,10) doesn't return 10");
	}
	i1[0]=9; i1[1]=9;
	if(d.abs_index(i1)!=99) {
		fail_test("dimensions::test_abs_index()", __FILE__, __LINE__,
			"abs(9,9) in (10,10) doesn't return 99");
	}

	index<4> i4a, i4b, i4c; i4b[0]=1; i4b[1]=4; i4b[2]=1; i4b[3]=13;
	index_range<4> ir4(i4a,i4b); dimensions<4> d4(ir4);
	d4.abs_index(154, i4c);
	if(i4c[0]!=1 || i4c[1]!=0 || i4c[2]!=1 || i4c[3]!=0) {
		fail_test("dimensions::test_abs_index()", __FILE__, __LINE__,
			"linear index 154 doesn't return (1,0,1,0) in "
			"(2,5,2,14)");
	}
}

void dimensions_test::test_comp() throw(libtest::test_exception) {
	index<2> i1a, i1b, i2b;
	i1b[0] = 1; i1b[1] = 2;
	i1b[0] = 2; i1b[1] = 3;
	index_range<2> ir1(i1a, i1b), ir2(i1a,i2b); // Indexes run from (0,0) to (1,2)
	dimensions<2> d1(ir1), d2(d1), d3(ir2);


	if(!(d1==d2)) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Equal comparison of identical dimensions returned false");
	}
	if(d1==d3) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Equal comparison of different dimensions returned true");
	}
	if(d1!=d2) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Unequal comparison of identical dimesions returned true");
	}

	if(!(d1!=d3)) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Unequal comparison of different dimesions returned false");
	}
}



} // namespace libtensor

