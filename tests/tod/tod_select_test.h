#ifndef LIBTENSOR_TOD_SELECT_TEST_H
#define LIBTENSOR_TOD_SELECT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::tod_select class

    \ingroup libtensor_tests_tod
**/
class tod_select_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    template<typename ComparePolicy>
    void test_1(size_t n, double c) throw(libtest::test_exception);

    template<typename ComparePolicy>
    void test_2(size_t n, double c) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SELECT_TEST_H
