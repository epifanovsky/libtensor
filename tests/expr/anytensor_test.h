#ifndef LIBTENSOR_ANYTENSOR_TEST_H
#define LIBTENSOR_ANYTENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::anytensor class

    \ingroup libtensor_expr_tests
**/
class anytensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_ANYTENSOR_TEST_H

