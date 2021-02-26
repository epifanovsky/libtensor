#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_vmpriority.h>

using namespace libtensor;


int main() {

    libtensor::index<4> i1, i2;
    i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    dense_tensor<4, double, allocator> t(dim);

    to_vmpriority<4, double>(t).set_priority();
    to_vmpriority<4, double>(t).unset_priority();
}


