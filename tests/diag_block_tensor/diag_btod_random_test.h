#ifndef LIBTENSOR_DIAG_BTOD_RANDOM_TEST_H
#define LIBTENSOR_DIAG_BTOD_RANDOM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_btod_random class

    \ingroup libtensor_diag_block_tensor_tests
**/
class diag_btod_random_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_RANDOM_TEST_H
