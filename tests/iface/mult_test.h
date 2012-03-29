#ifndef LIBTENSOR_MULT_TEST_H
#define    LIBTENSOR_MULT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::mult function

    \ingroup libtensor_tests_iface
**/
class mult_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_tt_1a() throw(libtest::test_exception);
    void test_tt_1b() throw(libtest::test_exception);
    void test_tt_2() throw(libtest::test_exception);
    void test_tt_3() throw(libtest::test_exception);
    void test_tt_4() throw(libtest::test_exception);
    void test_tt_5() throw(libtest::test_exception);
    void test_tt_6a() throw(libtest::test_exception);
    void test_tt_6b() throw(libtest::test_exception);
    void test_te_1() throw(libtest::test_exception);
    void test_te_2() throw(libtest::test_exception);
    void test_te_3() throw(libtest::test_exception);
    void test_et_1() throw(libtest::test_exception);
    void test_et_2() throw(libtest::test_exception);
    void test_et_3() throw(libtest::test_exception);
    void test_ee_1a() throw(libtest::test_exception);
    void test_ee_1b() throw(libtest::test_exception);
    void test_ee_2() throw(libtest::test_exception);

};

// tt mult / div
// tt perm 1 / 2
// tt scaled 1 / 2, div / mult

} // namespace libtensor

#endif // LIBTENSOR_MULT_TEST_H
