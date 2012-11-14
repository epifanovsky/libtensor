#ifndef LIBTENSOR_SHORT_ORBIT_TEST_H
#define LIBTENSOR_SHORT_ORBIT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::short_orbit class

    \ingroup libtensor_tests_core
**/
class short_orbit_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();

};


} // namespace libtensor

#endif // LIBTENSOR_SHORT_ORBIT_TEST_H
