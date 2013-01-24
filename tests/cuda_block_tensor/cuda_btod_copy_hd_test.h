#ifndef LIBTENSOR_CUDA_BTOD_COPY_HD_TEST_H
#define LIBTENSOR_CUDA_BTOD_COPY_HD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::cuda_btod_copy_dh class

    \ingroup libtensor_tests_btod
 **/
class cuda_btod_copy_hd_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test1() throw(libtest::test_exception);


};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_COPY_HD_TEST_H
