#ifndef LIBTENSOR_LETTER_EXPR_TEST_H
#define LIBTENSOR_LETTER_EXPR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::letter_expr class

    \ingroup libtensor_expr_tests
 **/
class letter_expr_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_contains() throw(libtest::test_exception);
    void test_permutation() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LETTER_EXPR_TEST_H
