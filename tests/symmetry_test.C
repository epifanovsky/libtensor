#include <libtensor.h>
#include "symmetry.h"
#include "symmetry_test.h"

namespace libtensor {

void symmetry_test::perform() throw(libtest::test_exception) {
	index<2> i1, i2;
	i2[0]=3; i2[1]=2;
	index_range<2> ir(i1,i2);
	dimensions<2> d(ir);
}

} // namespace libtensor

