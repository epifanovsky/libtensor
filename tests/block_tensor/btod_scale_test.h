#ifndef LIBTENSOR_BTOD_SCALE_TEST_H
#define LIBTENSOR_BTOD_SCALE_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/block_tensor/block_tensor_i.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_scale class

    \ingroup libtensor_tests_btod
**/
class btod_scale_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    template<size_t N>
    void test_generic(const char *testname,
        block_tensor_i<N, double> &bt, double c)
       ;

    void test_0();
    void test_i(size_t i);

    void test_1();
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SCALE_TEST_H
