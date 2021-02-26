#ifndef LIBTENSOR_EXPR_NEW_TEST_H
#define LIBTENSOR_EXPR_NEW_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests various problematic expressions

    \ingroup libtensor_tests_iface
 **/
template<typename T>
class expr_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
    static const T k_thresh; //!< Threshold multiplier

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);
    void test_3() throw(libtest::test_exception);
    void test_4() throw(libtest::test_exception);
    void test_5() throw(libtest::test_exception);
    void test_6() throw(libtest::test_exception);
    void test_7() throw(libtest::test_exception);
    void test_8() throw(libtest::test_exception);
    void test_9() throw(libtest::test_exception);
    void test_10() throw(libtest::test_exception);
    void test_11() throw(libtest::test_exception);
    void test_12() throw(libtest::test_exception);
    void test_13() throw(libtest::test_exception);
    void test_14() throw(libtest::test_exception);

};

class expr_test_new : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_EXPR_NEW_TEST_H
