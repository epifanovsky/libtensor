#include "bispace_1d.h"
#include "bispace_test.h"

namespace libtensor {

void bispace_test::perform() throw(libtest::test_exception) {
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
	dimensions<2> ij_dims(ij.dims());
	if(ij_dims[0]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 0 in ij");
	}
	if(ij_dims[1]!=10) {
		fail_test("bispace_test::perform()", __FILE__, __LINE__,
			"Incorrect dimension 1 in ij");
	}

	bispace<4> ijab((i&j)*(a&b));
	dimensions<4> ijab_dims(ijab.dims());
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
	dimensions<4> iajb_dims(iajb.dims());
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
	dimensions<1> iajb_dims_i(iajb[0].dims());
	dimensions<1> iajb_dims_a(iajb[1].dims());
	dimensions<1> iajb_dims_j(iajb[2].dims());
	dimensions<1> iajb_dims_b(iajb[3].dims());
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

} // namespace libtensor

