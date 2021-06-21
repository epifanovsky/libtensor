#ifndef LIBTENSOR_DIAG_TEST_H
#define    LIBTENSOR_DIAG_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::diag function

    \ingroup libtensor_tests_iface
 **/
class diag_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_t_1();
    void test_t_2();
    void test_t_3();
    void test_t_4();
    void test_t_5();
    void test_t_6();
    void test_t_7();
    void test_e_1();
    void test_x_1();

};

} // namespace libtensor

#endif // LIBTENSOR_DIAG_TEST_H
