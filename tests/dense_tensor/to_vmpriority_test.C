#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_vmpriority.h>
#include "to_vmpriority_test.h"

namespace libtensor {

void to_vmpriority_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_vmpriority_test_x<double>   ";
    to_vmpriority_test_x<double> t_d;
    t_d.perform();
    std::cout << "Testing to_vmpriority_test_x<float>   ";
    to_vmpriority_test_x<float> t_f;
    t_f.perform();
}

template<typename T>
void to_vmpriority_test_x<T>::perform() throw(libtest::test_exception) {

    typedef allocator<T> allocator_t;

    index<4> i1, i2;
    i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    dense_tensor<4, T, allocator_t> t(dim);

    to_vmpriority<4, T>(t).set_priority();
    to_vmpriority<4, T>(t).unset_priority();
}


} // namespace libtensor

