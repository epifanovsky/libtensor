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
}

} // namespace libtensor

