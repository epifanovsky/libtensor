#ifndef LIBTENSOR_DIRSUM_TEST_H
#define	LIBTENSOR_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::dirsum function

    \ingroup libtensor_expr_tests
 **/
class dirsum_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_tt_1() throw(libtest::test_exception);
    void test_tt_2() throw(libtest::test_exception);
    void test_tt_3() throw(libtest::test_exception);
    void test_tt_4() throw(libtest::test_exception);
    void test_tt_5() throw(libtest::test_exception);
    void test_tt_6() throw(libtest::test_exception);
    void test_te_1() throw(libtest::test_exception);
    void test_et_1() throw(libtest::test_exception);
    void test_ee_1() throw(libtest::test_exception);
    void test_ee_2() throw(libtest::test_exception);
    void test_ee_3() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_DIRSUM_TEST_H
