#ifndef LIBTENSOR_BTOD_SHIFT_DIAG_TEST_H
#define LIBTENSOR_BTOD_SHIFT_DIAG_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/block_index_space.h>
#include <libtensor/core/symmetry.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_shift_diag class

    \ingroup libtensor_tests_btod
**/
class btod_shift_diag_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);
    void test_3() throw(libtest::test_exception);
    void test_4() throw(libtest::test_exception);
    void test_5() throw(libtest::test_exception);
    void test_6() throw(libtest::test_exception);

    template<size_t N>
    void test_generic(const char *testname,
        const block_index_space<N> &bis, const symmetry<N, double> &sym,
        const sequence<N, size_t> &msk, double d)
        throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SHIFT_DIAG_TEST_H
