#ifndef LIBTENSOR_TOD_SET_DIAG_TEST_H
#define LIBTENSOR_TOD_SET_DIAG_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/dimensions.h>

namespace libtensor {

/** \brief Tests the libtensor::tod_set_diag class

    \ingroup libtensor_tests_tod
**/
class tod_set_diag_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    template<size_t N>
    void run_test1(const dimensions<N> &dims, double d, bool zero)
        throw(libtest::test_exception);

    template<size_t N>
    void run_test2(const dimensions<N> &dims, const sequence<N, size_t> &msk,
        double d, bool zero) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_DIAG_TEST_H
