#ifndef LIBTENSOR_LETTER_EXPR_TEST_H
#define LIBTENSOR_LETTER_EXPR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::letter_expr class

    \ingroup libtensor_tests_iface
**/
class letter_expr_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_contains();
    void test_permutation();

};


} // namespace libtensor

#endif // LIBTENSOR_LETTER_EXPR_TEST_H

