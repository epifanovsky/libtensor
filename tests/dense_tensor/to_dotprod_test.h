#ifndef LIBTENSOR_TO_DOTPROD_TEST_H
#define LIBTENSOR_TO_DOTPROD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::tod_dotprod class

    \ingroup libtensor_tests_tod
 **/
template <typename T>
class to_dotprod_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
    static const T k_thresh; //!< Threshold multiplier

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

class to_dotprod_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_DOTPROD_TEST_H

