#ifndef LIBTENSOR_SO_DIRSUM_TEST_H
#define LIBTENSOR_SO_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::so_dirsum class

    \ingroup libtensor_tests_sym
**/
class so_dirsum_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_empty_1() throw(libtest::test_exception);
    void test_empty_2() throw(libtest::test_exception);
    void test_empty_3() throw(libtest::test_exception);
    void test_se_1(bool s1, bool s2) throw(libtest::test_exception);
    void test_se_2() throw(libtest::test_exception);
    void test_se_3() throw(libtest::test_exception);
    void test_se_4() throw(libtest::test_exception);
    void test_perm_1() throw(libtest::test_exception);
    void test_perm_2() throw(libtest::test_exception);
    void test_vac_1() throw(libtest::test_exception);
    void test_vac_2() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_TEST_H
