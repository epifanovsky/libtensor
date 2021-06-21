#ifndef LIBTENSOR_SE_PART_TEST_H
#define LIBTENSOR_SE_PART_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::se_part class

    \ingroup libtensor_tests_sym
 **/
class se_part_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3a();
    void test_3b();
    void test_4();
    void test_5();
    void test_6();
    void test_perm_1();
    void test_perm_2();
    void test_perm_3();
    void test_perm_4();
    void test_perm_5();
    void test_exc();
};

} // namespace libtensor

#endif // LIBTENSOR_SE_PART_TEST_H

