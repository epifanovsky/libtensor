#ifndef LIBTENSOR_COMBINED_ORBITS_TEST_H
#define LIBTENSOR_COMBINED_ORBITS_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::combined_orbits class

    \ingroup libtensor_tests_core
**/
class combined_orbits_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();

};


} // namespace libtensor

#endif // LIBTENSOR_COMBINED_ORBITS_TEST_H
