#ifndef LIBTENSOR_BTOD_SET_DIAG_TEST_H
#define LIBTENSOR_BTOD_SET_DIAG_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/block_index_space.h>
#include <libtensor/core/symmetry.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_set_diag class

    \ingroup libtensor_tests_btod
**/
class btod_set_diag_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();

    template<size_t N>
    void test_generic(const char *testname, const block_index_space<N> &bis,
        const symmetry<N, double> &sym, double d)
       ;

    template<size_t N>
    void test_generic(const char *testname,
        const block_index_space<N> &bis, const symmetry<N, double> &sym,
        const sequence<N, size_t> &msk, double d)
       ;
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_DIAG_TEST_H
