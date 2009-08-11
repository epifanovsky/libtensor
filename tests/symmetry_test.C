#include <libtensor.h>
#include "symmetry_test.h"

namespace libtensor {

void symmetry_test::perform() throw(libtest::test_exception) {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	dimensions<2> dims(index_range<2>(i1, i2));
	symmetry<2, double> sym(dims);
}

} // namespace libtensor
