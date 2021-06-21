#ifndef LIBTENSOR_SO_DIRSUM_TEST_H
#define LIBTENSOR_SO_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::so_dirsum class

    \ingroup libtensor_tests_sym
**/
class so_dirsum_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_empty_1();
    void test_empty_2();
    void test_empty_3();
    void test_se_1(bool s1, bool s2);
    void test_se_2(bool s1, bool s2);
    void test_se_3();
    void test_se_4();
    void test_perm_1();
    void test_perm_2();
    void test_vac_1();
    void test_vac_2();
};

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_TEST_H
