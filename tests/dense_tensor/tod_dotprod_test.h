#ifndef LIBTENSOR_TOD_DOTPROD_TEST_H
#define LIBTENSOR_TOD_DOTPROD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::tod_dotprod class

    \ingroup libtensor_tests_tod
 **/
class tod_dotprod_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_i_i(size_t ni) throw(libtest::test_exception);
    void test_ij_ij(size_t ni, size_t nj) throw(libtest::test_exception);
    void test_ij_ji(size_t ni, size_t nj) throw(libtest::test_exception);
    void test_ijk_ijk(size_t ni, size_t nj, size_t nk)
        throw(libtest::test_exception);
    void test_ijk_ikj(size_t ni, size_t nj, size_t nk)
        throw(libtest::test_exception);
    void test_ijk_jik(size_t ni, size_t nj, size_t nk)
        throw(libtest::test_exception);
    void test_ijk_jki(size_t ni, size_t nj, size_t nk)
        throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_TEST_H

