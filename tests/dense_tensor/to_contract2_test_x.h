#ifndef LIBTENSOR_TO_CONTRACT2_TEST_H
#define LIBTENSOR_TO_CONTRACT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_contract2 class

    \ingroup libtensor_tests_tod
**/
template<typename T>
class to_contract2_test_x : public libtest::unit_test {
private:
    //static constexpr T k_thresh = 5e-14; //!< Threshold multiplier
    static const T k_thresh; //!< Threshold multiplier

public:
   virtual void perform() throw(libtest::test_exception);

private:
    // c = \sum_p a_p b_p
    void test_0_p_p(size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_i = \sum_p a_p b_{pi}
    void test_i_p_pi(size_t ni, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_i = \sum_p a_p b_{ip}
    void test_i_p_ip(size_t ni, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_i = \sum_p a_{pi} b_p
    void test_i_pi_p(size_t ni, size_t np, T d= 0.0)
        throw(libtest::test_exception);

    // c_i = \sum_p a_{ip} b_p
    void test_i_ip_p(size_t ni, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = c_{ij} + d a_i b_j
    void test_ij_i_j(size_t ni, size_t nj, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = c_{ij} + d a_j b_i
    
    void test_ij_j_i(size_t ni, size_t nj, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{pi} b_{pj}
    
    void test_ij_pi_pj(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{pi} b_{jp}
    
    void test_ij_pi_jp(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{ip} b_{pj}
    
    void test_ij_ip_pj(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{ip} b_{jp}
    
    void test_ij_ip_jp(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{pj} b_{pi}
    
    void test_ij_pj_pi(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{pj} b_{ip}
    
    void test_ij_pj_ip(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{jp} b_{ip}
    
    void test_ij_jp_ip(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{jp} b_{pi}
    
    void test_ij_jp_pi(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{p} b_{pji}
    
    void test_ij_p_pji(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_p a_{pji} b_{p}
    
    void test_ij_pji_p(size_t ni, size_t nj, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_{pq} a_{pqi} b_{pjq}
    
    void test_ij_pqi_pjq(size_t ni, size_t nj, size_t np, size_t nq,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ij} = \sum_{pq} a_{ipq} b_{jqp}
    
    void test_ij_ipq_jqp(size_t ni, size_t nj, size_t np, size_t nq,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ij} = \sum_{pq} a_{jpq} b_{iqp}
    
    void test_ij_jpq_iqp(size_t ni, size_t nj, size_t np, size_t nq,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ij} = c_{ij} + d \sum_{pq} a_{jipq} b_{qp}
    
    void test_ij_jipq_qp(size_t ni, size_t nj, size_t np, size_t nq,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ij} = \sum_{pq} a_{pq} b_{ijpq}
    
    void test_ij_pq_ijpq(size_t ni, size_t nj, size_t np, size_t nq)
        throw(libtest::test_exception);

    // c_{ij} = c_{ij} + d \sum_{pq} a_{pq} b_{ijpq}
    
    void test_ij_pq_ijpq_a(size_t ni, size_t nj, size_t np, size_t nq,
        T d) throw(libtest::test_exception);

    // c_{ij} = \sum_p a^1_{pi} b^1_{pj} + \sum_q a^2_{qi} b^2_{jq}
    
    void test_ij_pi_pj_qi_jq(size_t ni, size_t nj, size_t np, size_t nq,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ij} = \sum_p a^1_{pi} b^1_{pj} + \sum_q a^2_{qi} b^2_{qj}
    
    void test_ij_pi_pj_qi_qj(size_t ni, size_t nj, size_t np, size_t nq,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = \sum_{pq} a_{ip} b_{pkj}
    
    void test_ijk_ip_pkj(size_t ni, size_t nj, size_t nk, size_t np,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = \sum_{pq} a_{pi} b_{pkj}
    
    void test_ijk_pi_pkj(size_t ni, size_t nj, size_t nk, size_t np,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = \sum_{pq} a_{pik} b_{pj}
    
    void test_ijk_pik_pj(size_t ni, size_t nj, size_t nk, size_t np,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = \sum_{pq} a_{pj} b_{ipk}
    
    void test_ijk_pj_ipk(size_t ni, size_t nj, size_t nk, size_t np,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = \sum_{pq} a_{pj} b_{pik}
    
    void test_ijk_pj_pik(size_t ni, size_t nj, size_t nk, size_t np,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = \sum_{pq} a_{pkj} b_{ip}
    
    void test_ijk_pkj_ip(size_t ni, size_t nj, size_t nk, size_t np,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = \sum_{pq} a_{pkj} b_{pi}
    
    void test_ijk_pkj_pi(size_t ni, size_t nj, size_t nk, size_t np,
        T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = c_{ijk} + d \sum_{pq} a_{kjpq} b_{iqp}
    
    void test_ijk_kjpq_iqp(size_t ni, size_t nj, size_t nk, size_t np,
        size_t nq, T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = c_{ijk} + d \sum_{pq} a_{pkiq} b_{pjq}
    
    void test_ijk_pkiq_pjq(size_t ni, size_t nj, size_t nk, size_t np,
        size_t nq, T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = c_{ijk} + d \sum_{pq} a_{pqj} b_{iqpk}
    
    void test_ijk_pqj_iqpk(size_t ni, size_t nj, size_t nk, size_t np,
        size_t nq, T d = 0.0) throw(libtest::test_exception);

    // c_{ijk} = c_{ijk} + d \sum_{pq} a_{pqji} b_{qpk}
    
    void test_ijk_pqji_qpk(size_t ni, size_t nj, size_t nk, size_t np,
        size_t nq, T d = 0.0) throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ikp} b_{jpl}
    
    void test_ijkl_ikp_jpl(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, T d = 0.0) throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ipk} b_{jpl}
    
    void test_ijkl_ipk_jpl(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, T d = 0.0) throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ipl} b_{jpk}
    
    void test_ijkl_ipl_jpk(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, T d = 0.0) throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{jkp} b_{ipl}
    
    void test_ijkl_jkp_ipl(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, T d = 0.0) throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{jpl} b_{ipk}
    
    void test_ijkl_jpl_ipk(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, T d = 0.0) throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{iplq} b_{kpjq}
    
    void test_ijkl_iplq_kpjq(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{iplq} b_{pkjq}
    
    void test_ijkl_iplq_pkjq(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{iplq} b_{pkqj}
    
    void test_ijkl_iplq_pkqj(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{ipql} b_{kpqj}
    
    void test_ijkl_ipql_kpqj(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{ipql} b_{pkqj}
    
    void test_ijkl_ipql_pkqj(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pilq} b_{kpjq}
    
    void test_ijkl_pilq_kpjq(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pilq} b_{pkjq}
    
    void test_ijkl_pilq_pkjq(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{piql} b_{kpqj}
    
    void test_ijkl_piql_kpqj(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{piql} b_{pkqj}
    
    void test_ijkl_piql_pkqj(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pqkj} b_{iqpl}
    
    void test_ijkl_pqkj_iqpl(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pqkj} b_{qipl}
    
    void test_ijkl_pqkj_qipl(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, size_t nq, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a^1_{jpl} b^1_{ipk}
    //                     + d \sum_{q} a^2_{jiq} b^2_{kql}
    //                     + d \sum_{r} a^3_{jlr} b^3_{ikr}
    
    void test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(size_t ni, size_t nj, size_t nk,
        size_t nl, size_t np, size_t nq, size_t nr, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ij} = \sum_{pqr} a_{ipqr} b_{jpqr}
    
    void test_ij_ipqr_jpqr(size_t ni, size_t nj, size_t np, size_t nq,
        size_t nr) throw(libtest::test_exception);

    // c_{ij} = c_{ij} + d \sum_{pqr} a_{ipqr} b_{jpqr}
    
    void test_ij_ipqr_jpqr_a(size_t ni, size_t nj, size_t np, size_t nq,
        size_t nr, T d) throw(libtest::test_exception);

    // c_{ij} = \sum_{pqr} a_{ipqr} b_{pjrq}
    
    void test_ij_ipqr_pjrq(size_t ni, size_t nj, size_t np, size_t nq,
        size_t nr, T d = 0.0) throw(libtest::test_exception);

    // c_{ij} = c_{ij} + d \sum_{pqr} a_{jpqr} b_{iprq}
    
    void test_ij_jpqr_iprq(size_t ni, size_t nj, size_t np, size_t nq,
        size_t nr, T d = 0.0) throw(libtest::test_exception);

    // c_{ij} = \sum_{pqr} a_{pqir} b_{pqjr}
    
    void test_ij_pqir_pqjr(size_t ni, size_t nj, size_t np, size_t nq,
        size_t nr) throw(libtest::test_exception);

    // c_{ij} = c_{ij} + d \sum_{pqr} a_{pqir} b_{pqjr}
    
    void test_ij_pqir_pqjr_a(size_t ni, size_t nj, size_t np, size_t nq,
        size_t nr, T d) throw(libtest::test_exception);

    // c_{ijkl} = \sum_{p} a_{pi} b_{jklp}
    
    void test_ijkl_pi_jklp(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np) throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{pi} b_{jklp}
    
    void test_ijkl_pi_jklp_a(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, T d) throw(libtest::test_exception);

    // c_{jikl} = \sum_{p} a_{pi} b_{jpkl}
    
    void test_jikl_pi_jpkl(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np) throw(libtest::test_exception);

    // c_{jikl} = c_{jikl} + d \sum_{p} a_{pi} b_{jpkl}
    
    void test_jikl_pi_jpkl_a(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, T d) throw(libtest::test_exception);

    // c_{ijkl} = \sum_{p} a_{ijp} b_{klp}
    
    void test_ijkl_ijp_klp(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np) throw(libtest::test_exception);

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ijp} b_{klp}
    
    void test_ijkl_ijp_klp_a(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t np, T d) throw(libtest::test_exception);

    // c_{ijkl} = a_{ij} b_{kl}
    void test_ijkl_ij_kl(size_t ni, size_t nj, size_t nk, size_t nl)
        throw(libtest::test_exception);

    // c_{ijkl} = a_{ij} b_{lk}
    void test_ijkl_ij_lk(size_t ni, size_t nj, size_t nk, size_t nl)
        throw(libtest::test_exception);

    // c_{ijklm} = c_{ijklm} + d \sum_{p} a_{ikp} b_{jpml}
    void test_ijklm_ikp_jpml(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t nm, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijklm} = c_{ijklm} + d \sum_{p} a_{ipkm} b_{jpl}
    void test_ijklm_ipkm_jpl(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t nm, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijklm} = c_{ijklm} + d \sum_{p} a_{jlp} b_{ipkm}
    void test_ijklm_jlp_ipkm(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t nm, size_t np, T d = 0.0)
        throw(libtest::test_exception);

    // c_{ijklmn} = c_{ijklmn} + d \sum_{p} a_{kjmp} b_{ipln}
    void test_ijklmn_kjmp_ipln(size_t ni, size_t nj, size_t nk, size_t nl,
        size_t nm, size_t nn, size_t np, T d = 0.0)
        throw(libtest::test_exception);

};


class to_contract2_test : public libtest::unit_test  {
public:
    virtual void perform() throw(libtest::test_exception);
};

/*
//class to_contract_test : public to_contract_test_x<double>, to_contract_test_x<float> {
class to_contract_test : public to_contract_test_x<double> {
public:
    virtual void perform() throw(libtest::test_exception);
};
*/



} // namespace libtensor

#endif // LIBTENSOR_TO_CONTRACT2_TEST_H

