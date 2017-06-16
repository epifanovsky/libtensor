#ifndef LIBTENSOR_TO_RANDOM_TEST_H
#define LIBTENSOR_TO_RANDOM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_random class

    \ingroup libtensor_tests_tod
 **/
template<typename T>
class to_random_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

};

class to_random_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TO_RANDOM_TEST_H

