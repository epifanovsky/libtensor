#ifndef LIBTENSOR_DIAG_BTOD_CONTRACT2_TEST_H
#define LIBTENSOR_DIAG_BTOD_CONTRACT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_btod_contract2 class

    \ingroup libtensor_diag_block_tensor_tests
 **/
class diag_btod_contract2_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_CONTRACT2_TEST_H
