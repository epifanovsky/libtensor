#ifndef LIBTENSOR_TO_CONTRACT2_PERMS_TEST_H
#define LIBTENSOR_TO_CONTRACT2_PERMS_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_contract2_perms class

    \ingroup libtensor_tests_tod
**/
class to_contract2_perms_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ij_i_j(size_t ni, size_t nj) throw(libtest::test_exception);
    void test_ij_j_i(size_t ni, size_t nj) throw(libtest::test_exception);
    void test_ij_ik_jk(size_t ni, size_t nj, size_t nk)
        throw(libtest::test_exception);
    void test_ijk_ij_k(size_t ni, size_t nj, size_t nk)
          throw(libtest::test_exception);

    void test_ijk_ji_k(size_t ni, size_t nj, size_t nk)
           throw(libtest::test_exception);

    void test_ijk_jil_kl(size_t ni, size_t nj, size_t nk, size_t nl)
        throw(libtest::test_exception);

    void test_ijab_ijkl_klab(size_t ni, size_t nj, size_t nk, size_t nl, size_t na, size_t nb)
        throw(libtest::test_exception);
    void test_ijab_lijk_klab(size_t ni, size_t nj, size_t nk, size_t nl, size_t na, size_t nb)
        throw(libtest::test_exception);
};


} // namespace libtensor

#endif // LIBTENSOR_TO_CONTRACT2_PERMS_TEST_H

