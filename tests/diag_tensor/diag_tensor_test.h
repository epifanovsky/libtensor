#ifndef LIBTENSOR_DIAG_TENSOR_TEST_H
#define LIBTENSOR_DIAG_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_tensor class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_tensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TENSOR_TEST_H

