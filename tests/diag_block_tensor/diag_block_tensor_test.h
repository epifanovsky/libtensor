#ifndef LIBTENSOR_DIAG_BLOCK_TENSOR_TEST_H
#define LIBTENSOR_DIAG_BLOCK_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_block_tensor class

    \ingroup libtensor_diag_block_tensor_tests
 **/
class diag_block_tensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BLOCK_TENSOR_TEST_H
