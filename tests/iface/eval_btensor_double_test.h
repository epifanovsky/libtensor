#ifndef LIBTENSOR_EVAL_BTENSOR_DOUBLE_TEST_H
#define LIBTENSOR_EVAL_BTENSOR_DOUBLE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::iface::eval_btensor<double> class

    \ingroup libtensor_tests_iface
**/
class eval_btensor_double_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_copy_1();
    void test_copy_2();
    void test_copy_3();
    void test_contract_1();

};


} // namespace libtensor

#endif // LIBTENSOR_EVAL_BTENSOR_DOUBLE_TEST_H

