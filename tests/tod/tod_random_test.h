#ifndef LIBTENSOR_TOD_RANDOM_TEST_H
#define LIBTENSOR_TOD_RANDOM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::tod_random class

    \ingroup libtensor_tests_tod
**/
class tod_random_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_RANDOM_TEST_H

