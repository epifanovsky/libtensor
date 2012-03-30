#ifndef LIBTENSOR_DIMENSIONS_TEST_H
#define LIBTENSOR_DIMENSIONS_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::dimensions class

    \ingroup libtensor_tests_core
 **/
class dimensions_test: public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ctor() throw(libtest::test_exception);
    void test_contains() throw(libtest::test_exception);
    void test_comp() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_DIMENSIONS_TEST_H
