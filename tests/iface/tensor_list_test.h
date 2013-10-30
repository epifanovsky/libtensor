#ifndef LIBTENSOR_TENSOR_LIST_TEST_H
#define LIBTENSOR_TENSOR_LIST_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::tensor_list class

    \ingroup libtensor_tests_iface
**/
class tensor_list_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();

};


} // namespace libtensor

#endif // LIBTENSOR_TENSOR_LIST_TEST_H

