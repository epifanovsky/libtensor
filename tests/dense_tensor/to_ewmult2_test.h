#ifndef LIBTENSOR_TO_EWMULT2_TEST_H
#define LIBTENSOR_TO_EWMULT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_ewmult2 class

    \ingroup libtensor_tests_tod
 **/
template<typename T>
class to_ewmult2_test_x : public libtest::unit_test {
private:
    static const T k_thresh; //!< Threshold multiplier

public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_i_i_i(size_t ni, T d = 0.0)
        throw(libtest::test_exception);
    void test_ij_ij_ij(size_t ni, size_t nj, T d = 0.0)
        throw(libtest::test_exception);
    void test_ij_ij_ji(size_t ni, size_t nj, T d = 0.0)
        throw(libtest::test_exception);
    void test_ijk_jki_kij(size_t ni, size_t nj, size_t nk, T d = 0.0)
        throw(libtest::test_exception);
    void test_ijk_ik_kj(size_t ni, size_t nj, size_t nk, T d = 0.0)
        throw(libtest::test_exception);
    void test_ijkl_kj_ikl(size_t ni, size_t nj, size_t nk, size_t nl,
        T d = 0.0) throw(libtest::test_exception);
    void test_ijkl_ljk_jil(size_t ni, size_t nj, size_t nk, size_t nl,
        T d = 0.0) throw(libtest::test_exception);

};

class to_ewmult2_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_EWMULT2_TEST_H
