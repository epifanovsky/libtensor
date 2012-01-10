#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/tod/tod_set.h>
#include "tod_set_test.h"

namespace libtensor {

typedef dense_tensor<4, double, std_allocator<double> > tensor4_d;

void tod_set_test::perform() throw(libtest::test_exception) {

    cpu_pool cpus(1);

	index<4> i1, i2;
	i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir);
	tensor4_d t(dim);

	tod_set<4> op(5.0);
	op.perform(cpus, t);
}

} // namespace libtensor

