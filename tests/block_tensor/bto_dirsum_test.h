#ifndef LIBTENSOR_BTO_DIRSUM_TEST_H
#define LIBTENSOR_BTO_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::bto_dirsum class

    \ingroup libtensor_tests_btod
**/
template<typename T>
class bto_dirsum_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
    static const T k_thresh; //!< Threshold multiplier

private:
    // c_{ij} = a_i + b_j
    void test_ij_i_j_1(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = a_i + a_j (with more than 1 block, checking for symmetry)
    void test_ij_i_j_2(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = a_i - a_j (with more than 1 block, checking for symmetry)
    void test_ij_i_j_3(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijk} = a_{ij} + b_k
    void test_ijk_ij_k_1(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ikjl} = a_{ij} + b_{kl}
    void test_ikjl_ij_kl_1(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ikjl} = a_{ij} + b_{kl} (with more than 1 block)
    void test_ikjl_ij_kl_2(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ikjl} = a_{ij} + b_{kl} (with more than 1 block and se_perm)
    void test_ikjl_ij_kl_3a(bool s1, bool s2, bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ikjl} = a_{ij} + b_{kl} (with more than 1 block and se_part)
    void test_ikjl_ij_kl_3b(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ikjl} = a_{ij} + b_{kl} (with more than 1 block and se_label)
    void test_ikjl_ij_kl_3c(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // c_{iklj} = a_{ij} + a_{kl} (with more than 1 block and symmetry elements)
    void test_iklj_ij_kl_1(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

    // b_{ijkl} = a_{ij} + a_{kl}
    // c_{ikmjln} = b_{ijkl} + a_{mn}
    // (with more than 1 block and symmetry elements)
    void test_ikmjln_ij_kl_mn(bool rnd, T d = 0.0)
        throw(libtest::test_exception);

};

class bto_dirsum_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTO_DIRSUM_TEST_H
