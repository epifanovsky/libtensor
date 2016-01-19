#ifndef LIBTENSOR_BTOD_SYMCONTRACT3_TEST_H
#define LIBTENSOR_BTOD_SYMCONTRACT3_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::btod_symcontract3 class

    \ingroup libtensor_tests_btod
**/
class btod_symcontract3_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_contr_1();
    void test_contr_2();
    void test_contr_3();
    void test_contr_4();
    void test_contr_5();
    void test_contr_6();
    void test_contr_7();

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMCONTRACT3_TEST_H
