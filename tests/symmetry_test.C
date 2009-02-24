#include "default_symmetry.h"
#include "symmetry_test.h"

namespace libtensor {

void symmetry_test::perform() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0]=3; i2[1]=2;
	index_range ir(i1,i2);
	dimensions d(ir);
	default_symmetry ds(2);
	symmetry sym(ds, d);
}

} // namespace libtensor

