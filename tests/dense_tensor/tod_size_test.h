#ifndef LIBTENSOR_TOD_SIZE_TEST_H
#define LIBTENSOR_TOD_SIZE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::tod_size class

    \ingroup libtensor_tests_tod
**/
class tod_size_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SIZE_TEST_H

