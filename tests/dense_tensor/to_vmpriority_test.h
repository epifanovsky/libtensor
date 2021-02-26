#ifndef LIBTENSOR_TO_VMPRIORITY_TEST_H
#define LIBTENSOR_TO_VMPRIORITY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_vmpriority class

    \ingroup libtensor_tests_tod
 **/
template<typename T>
class to_vmpriority_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

};

class to_vmpriority_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_VMPRIORITY_TEST_H

