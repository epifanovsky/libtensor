#include <libtensor.h>
#include "bispace_test.h"

namespace libtensor {

void bispace_test::perform() throw(libtest::test_exception) {
	test_1();

	bispace<1> i(10), j(10), a(20), b(20);
	i.split(5);
	a.split(5).split(10).split(15);

	bispace<1> k(i), l(j), c(a), d(b);

	i&j;
	i&j&k;
	i&j&k&l;
	(i&j)&(k&l);

	i*j;
	i*j*k;
	i*j*a;
	(i&j)*k;

	bispace<2> ij(i&j);
	dimensions<2> ij_dims(ij.get_dims());
	if(ij_dims[0]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 0 in ij");
	}
	if(ij_dims[1]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 1 in ij");
	}

	bispace<4> ijab((i&j)*(a&b));
	dimensions<4> ijab_dims(ijab.get_dims());
	if(ijab_dims[0]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 0 in ijab");
	}
	if(ijab_dims[1]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 1 in ijab");
	}
	if(ijab_dims[2]!=20) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 2 in ijab");
	}
	if(ijab_dims[3]!=20) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 3 in ijab");
	}

	bispace<4> iajb(i*a*j*b, (i&j)*(a&b));
	dimensions<4> iajb_dims(iajb.get_dims());
	if(iajb_dims[0]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 0 in iajb");
	}
	if(iajb_dims[1]!=20) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 1 in iajb");
	}
	if(iajb_dims[2]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 2 in iajb");
	}
	if(iajb_dims[3]!=20) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 3 in iajb");
	}
	dimensions<1> iajb_dims_i(iajb[0].get_dims());
	dimensions<1> iajb_dims_a(iajb[1].get_dims());
	dimensions<1> iajb_dims_j(iajb[2].get_dims());
	dimensions<1> iajb_dims_b(iajb[3].get_dims());
	if(iajb_dims_i[0]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect single dimension i in iajb");
	}
	if(iajb_dims_a[0]!=20) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect single dimension a in iajb");
	}
	if(iajb_dims_j[0]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect single dimension j in iajb");
	}
	if(iajb_dims_b[0]!=20) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect single dimension b in iajb");
	}
}

void bispace_test::test_1() throw(libtest::test_exception) {

	const char *testname = "bispace_test::test_1()";

	try {

	index<1> i0, i1, i2;
	i2[0] = 2;
	dimensions<1> dim3(index_range<1>(i1, i2));
	i2[0] = 4;
	dimensions<1> dim5(index_range<1>(i1, i2));
	i2[0] = 5;
	dimensions<1> dim6(index_range<1>(i1, i2));
	i2[0] = 8;
	dimensions<1> dim9(index_range<1>(i1, i2));
	i2[0] = 19;
	dimensions<1> dim20(index_range<1>(i1, i2));

	bispace<1> a(20);
	a.split(5).split(11);
	const block_index_space<1> &bis = a.get_bis();

	if(!bis.get_dims().equals(dim20)) {
		fail_test(testname, __FILE__, __LINE__,
			"Total dimensions don't match reference");
	}
	if(!bis.get_block_index_dims().equals(dim3)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block index dimensions don't match reference");
	}
	i1[0] = 1; i2[0] = 2;
	if(!bis.get_block_dims(i0).equals(dim5)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block [0] dimensions don't match reference");
	}
	if(!bis.get_block_dims(i1).equals(dim6)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block [0] dimensions don't match reference");
	}
	if(!bis.get_block_dims(i2).equals(dim9)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block [0] dimensions don't match reference");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}

} // namespace libtensor

