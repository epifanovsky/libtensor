#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_vmpriority.h>
#include "tod_vmpriority_test.h"

namespace libtensor {


void tod_vmpriority_test::perform() throw(libtest::test_exception) {

    typedef std_allocator<double> allocator_t;

    index<4> i1, i2;
    i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    dense_tensor<4, double, allocator_t> t(dim);

    tod_vmpriority<4>(t).set_priority();
    tod_vmpriority<4>(t).unset_priority();
}


} // namespace libtensor

