#ifndef LIBTENSOR_TRACE_TEST_H
#define LIBTENSOR_TRACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::trace function

    \ingroup libtensor_tests_iface
 **/
class trace_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_t_1() throw(libtest::test_exception);
    void test_t_2() throw(libtest::test_exception);
    void test_e_1() throw(libtest::test_exception);
    void test_e_2() throw(libtest::test_exception);
    void test_e_3() throw(libtest::test_exception);

    void check_ref(const char *testname, double d, double d_ref)
        throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TRACE_TEST_H
