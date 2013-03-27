#ifndef LIBTENSOR_CUDA_TOD_SET_TEST_H
#define LIBTENSOR_CUDA_TOD_SET_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/cuda/cuda_allocator.h>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>

namespace libtensor {

/** \brief Tests the libtensor::cuda_tod_set class

    \ingroup libtensor_tests_tod
 **/
class cuda_tod_set_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1(double v);
    void test_2(double v);

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_TOD_SET_TEST_H

