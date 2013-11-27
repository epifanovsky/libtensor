#ifndef LIBTENSOR_DIAG_TOD_RANDOM_TEST_H
#define LIBTENSOR_DIAG_TOD_RANDOM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_tod_random class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_tod_random_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_RANDOM_TEST_H

