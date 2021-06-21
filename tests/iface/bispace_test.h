#ifndef LIBTENSOR_BISPACE_TEST_H
#define    LIBTENSOR_BISPACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::bispace class

    \ingroup libtensor_tests_iface
 **/
class bispace_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();
    void test_7();
    void test_8();
    void test_9();

};

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_TEST_H

