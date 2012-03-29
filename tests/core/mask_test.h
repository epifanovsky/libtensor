#ifndef LIBTENSOR_MASK_TEST_H
#define LIBTENSOR_MASK_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::mask class

    \ingroup libtensor_tests_core
**/
class mask_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_op_1() throw(libtest::test_exception);
    void test_op_2() throw(libtest::test_exception);
    void test_op_3() throw(libtest::test_exception);
    void test_op_4() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_MASK_TEST_H
