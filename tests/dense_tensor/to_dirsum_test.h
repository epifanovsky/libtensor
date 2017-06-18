#ifndef LIBTENSOR_TO_DIRSUM_TEST_H
#define LIBTENSOR_TO_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_dirsum class

    \ingroup libtensor_tests_tod
 **/
template <typename T>
class to_dirsum_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    // c_{ij} = a_i + b_j
    void test_ij_i_j_1(size_t ni, size_t nj, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = a_i - b_j
    void test_ij_i_j_2(size_t ni, size_t nj, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ikj} = a_{ij} + b_k
    void test_ikj_ij_k_1(size_t ni, size_t nj, size_t nk,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ikjl} = a_{ij} + b_{kl}
    void test_ikjl_ij_kl_1(size_t ni, size_t nj, size_t nk, size_t nl,
        T d = 0.0) throw(libtest::test_exception);

    // c_{iklj} = a_{ij} + b_{kl}
    void test_iklj_ij_kl_1(size_t ni, size_t nj, size_t nk, size_t nl,
        T d = 0.0) throw(libtest::test_exception);

};

class to_dirsum_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_DIRSUM_TEST_H

