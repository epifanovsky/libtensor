#ifndef LIBTENSOR_SUBGROUP_ORBITS_TEST_H
#define LIBTENSOR_SUBGROUP_ORBITS_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::subgroup_orbits class

    \ingroup libtensor_tests_core
**/
class subgroup_orbits_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();

};


} // namespace libtensor

#endif // LIBTENSOR_SUBGROUP_ORBITS_TEST_H
