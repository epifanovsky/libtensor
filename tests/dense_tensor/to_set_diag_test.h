#ifndef LIBTENSOR_TO_SET_DIAG_TEST_H
#define LIBTENSOR_TO_SET_DIAG_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/dimensions.h>

namespace libtensor {

/** \brief Tests the libtensor::to_set_diag class

    \ingroup libtensor_tests_tod
**/
template<typename T>
class to_set_diag_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    template<size_t N>
    void run_test1(const dimensions<N> &dims, T d, bool zero)
        throw(libtest::test_exception);

    template<size_t N>
    void run_test2(const dimensions<N> &dims, const sequence<N, size_t> &msk,
        T d, bool zero) throw(libtest::test_exception);
};

class to_set_diag_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_SET_DIAG_TEST_H
