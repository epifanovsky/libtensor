#ifndef LIBTENSOR_LABELED_ANYTENSOR_TEST_H
#define	LIBTENSOR_LABELED_ANYTENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::labeled_anytensor class

    \ingroup libtensor_expr_tests
**/
class labeled_anytensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_label() throw(libtest::test_exception);
    void test_expr_int() throw(libtest::test_exception);
    void test_expr_double() throw(libtest::test_exception);
    void test_expr_complex_1() throw(libtest::test_exception);
    void test_expr_complex_2() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LABELED_ANYTENSOR_TEST_H
