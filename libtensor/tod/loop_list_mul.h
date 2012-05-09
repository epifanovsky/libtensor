#ifndef LIBTENSOR_LOOP_LIST_MUL_H
#define LIBTENSOR_LOOP_LIST_MUL_H

#include <list>
#include "../timings.h"
#include "loop_list_base.h"

namespace libtensor {


/** \brief Operates nested loops on two arrays with multiplication and
        accumulation as the kernel (c += a * b)

    \ingroup libtensor_tod
 **/
class loop_list_mul :
    public loop_list_base<2, 1, loop_list_mul>,
    public timings<loop_list_mul> {

public:
    static const char *k_clazz; //!< Class name

private:
    struct {
        double m_d;
        size_t m_n;
        size_t m_stepa, m_stepb, m_stepc;
    } m_generic;

    //!    c = a_p b_p
    struct args_x_p_p {
        double d;
        size_t np;
        size_t spa, spb;
    } m_x_p_p;

    //!    c = a_pq b_qp
    struct args_x_pq_qp {
        double d;
        size_t np, nq;
        size_t spa, sqb;
    } m_x_pq_qp;

    //!    c_i = a_i b
    struct args_i_i_x {
        double d;
        size_t ni;
        size_t sic;
    } m_i_i_x;

    //!    c_i = a b_i
    struct args_i_x_i {
        double d;
        size_t ni;
        size_t sic;
    } m_i_x_i;

    //!    c_i = a_ip b_p
    struct args_i_ip_p {
        double d;
        size_t ni, np;
        size_t sia, sic, spb;
    } m_i_ip_p;

    //!    c_i = a_pi b_p
    struct args_i_pi_p {
        double d;
        size_t ni, np;
        size_t sic, spa, spb;
    } m_i_pi_p;

    //!    c_i = a_p b_ip
    struct args_i_p_ip {
        double d;
        size_t ni, np;
        size_t sib, sic, spa;
    } m_i_p_ip;

    //!    c_i = a_p b_pi
    struct args_i_p_pi {
        double d;
        size_t ni, np;
        size_t sic, spa, spb;
    } m_i_p_pi;

    //!    c_ij = a_ip b_pj
    struct args_ij_ip_pj {
        double d;
        size_t ni, nj, np;
        size_t sia, sic, spb;
    } m_ij_ip_pj;

    //!    c_ij = a_ip b_jp
    struct args_ij_ip_jp {
        double d;
        size_t ni, nj, np;
        size_t sia, sic, sjb;
    } m_ij_ip_jp;

    //!    c_ij = a_pi b_pj
    struct args_ij_pi_pj {
        double d;
        size_t ni, nj, np;
        size_t sic, spa, spb;
    } m_ij_pi_pj;

    //!    c_ij = a_pi b_jp
    struct args_ij_pi_jp {
        double d;
        size_t ni, nj, np;
        size_t sic, sjb, spa;
    } m_ij_pi_jp;

    //!    c_ij = a_pj b_ip
    struct args_ij_pj_ip {
        double d;
        size_t ni, nj, np;
        size_t sib, sic, spa;
    } m_ij_pj_ip;

    //!    c_ij = a_jp b_ip
    struct args_ij_jp_ip {
        double d;
        size_t ni, nj, np;
        size_t sib, sic, sja;
    } m_ij_jp_ip;

    //!    c_ij = a_pj b_pi
    struct args_ij_pj_pi {
        double d;
        size_t ni, nj, np;
        size_t sic, spa, spb;
    } m_ij_pj_pi;

    //!    c_ij = a_jp b_pi
    struct args_ij_jp_pi {
        double d;
        size_t ni, nj, np;
        size_t sic, sja, spb;
    } m_ij_jp_pi;

    //!    c_i = a_ipq b_qp
    struct args_i_ipq_qp {
        double d;
        size_t ni, np, nq;
        size_t sia, sic, spa, sqb;
    } m_i_ipq_qp;

    //!    c_i = a_pq b_iqp
    struct args_i_pq_iqp {
        double d;
        size_t ni, np, nq;
        size_t sib, sic, spa, sqb;
    } m_i_pq_iqp;

    //!    c_ij = a_ipq b_jqp
    struct args_ij_ipq_jqp {
        double d;
        size_t ni, nj, np, nq;
        size_t sia, sic, sjb, spa, sqb;
    } m_ij_ipq_jqp;

    //!    c_ij = a_jpq b_iqp
    struct args_ij_jpq_iqp {
        double d;
        size_t ni, nj, np, nq;
        size_t sib, sic, sja, spa, sqb;
    } m_ij_jpq_iqp;

    //!    c_ijkl = a_iplq b_kpjq
    struct args_ijkl_iplq_kpjq {
        double d;
        size_t ni, nj, nk, nl, np, nq;
    } m_ijkl_iplq_kpjq;

    //!    c_ijkl = a_iplq b_pkjq
    struct args_ijkl_iplq_pkjq {
        double d;
        size_t ni, nj, nk, nl, np, nq;
    } m_ijkl_iplq_pkjq;

    //!    c_ijkl = a_iplq b_pkqj
    struct args_ijkl_iplq_pkqj {
        double d;
        size_t ni, nj, nk, nl, np, nq;
    } m_ijkl_iplq_pkqj;

    //!    c_ijkl = a_ipql b_pkqj
    struct args_ijkl_ipql_pkqj {
        double d;
        size_t ni, nj, nk, nl, np, nq;
    } m_ijkl_ipql_pkqj;

    //!    c_ijkl = a_pilq b_kpjq
    struct args_ijkl_pilq_kpjq {
        double d;
        size_t ni, nj, nk, nl, np, nq;
    } m_ijkl_pilq_kpjq;

    //!    c_ijkl = a_pilq b_pkjq
    struct args_ijkl_pilq_pkjq {
        double d;
        size_t ni, nj, nk, nl, np, nq;
    } m_ijkl_pilq_pkjq;

    //!    c_ijkl = a_piql b_kpqj
    struct args_ijkl_piql_kpqj {
        double d;
        size_t ni, nj, nk, nl, np, nq;
    } m_ijkl_piql_kpqj;

    //!    c_ijkl = a_piql b_pkqj
    struct args_ijkl_piql_pkqj {
        double d;
        size_t ni, nj, nk, nl, np, nq;
    } m_ijkl_piql_pkqj;

    const char *m_kernelname;

protected:
    void run_loop(list_t &loop, registers &r, double c);

private:
    void match_l1(list_t &loop, double d);
    void match_x_p_p(list_t &loop, double d, size_t np, size_t spa);
    void match_i_i_x(list_t &loop, double d, size_t ni, size_t k1);
    void match_i_x_i(list_t &loop, double d, size_t ni, size_t k1);
    void match_dgemv_n_a_l3(list_t &loop, double d, size_t w1, size_t w2,
        size_t k1w1);
    void match_dgemv_n_b_l3(list_t &loop, double d, size_t w1, size_t w2,
        size_t k1w1);
    void match_dgemv_t_a1_l3(list_t &loop, double d, size_t w1, size_t w2,
        size_t k1, size_t k2w1);
    void match_dgemv_t_a2_l3(list_t &loop, double d, size_t w1, size_t w2,
        size_t k2w1, size_t k3);
    void match_dgemv_t_b1_l3(list_t &loop, double d, size_t w1, size_t w2,
        size_t k1, size_t k2w1);
    void match_dgemv_t_b2_l3(list_t &loop, double d, size_t w1, size_t w2,
        size_t k2w1, size_t k3);
    void match_x_pq_qp(list_t &loop, double d, size_t np, size_t nq,
        size_t spa, size_t sqb);
    void match_i_ipq_qp(list_t &loop, double d, size_t ni, size_t np,
        size_t nq, size_t sia, size_t spa, size_t sqb);
    void match_i_pq_iqp(list_t &loop, double d, size_t ni, size_t np,
        size_t nq, size_t sib, size_t spa, size_t sqb);
    void match_ij_jp_ip(list_t &loop);
    void match_ij_jp_pi(list_t &loop);
    void match_ij_pj_pi(list_t &loop);

    void fn_generic(registers &r) const;
    void fn_x_p_p(registers &r) const;
    void fn_x_pq_qp(registers &r) const;
    void fn_i_i_x(registers &r) const;
    void fn_i_x_i(registers &r) const;
    void fn_i_ip_p(registers &r) const;
    void fn_i_pi_p(registers &r) const;
    void fn_i_p_ip(registers &r) const;
    void fn_i_p_pi(registers &r) const;
    void fn_ij_ip_pj(registers &r) const;
    void fn_ij_ip_jp(registers &r) const;
    void fn_ij_pi_pj(registers &r) const;
    void fn_ij_pi_jp(registers &r) const;
    void fn_ij_pj_ip(registers &r) const;
    void fn_ij_jp_ip(registers &r) const;
    void fn_ij_pj_pi(registers &r) const;
    void fn_ij_jp_pi(registers &r) const;
    void fn_i_ipq_qp(registers &r) const;
    void fn_i_pq_iqp(registers &r) const;
    void fn_ij_ipq_jqp(registers &r) const;
    void fn_ij_jpq_iqp(registers &r) const;
    void fn_ijkl_iplq_kpjq(registers &r) const;
    void fn_ijkl_iplq_pkjq(registers &r) const;
    void fn_ijkl_iplq_pkqj(registers &r) const;
    void fn_ijkl_ipql_pkqj(registers &r) const;
    void fn_ijkl_pilq_kpjq(registers &r) const;
    void fn_ijkl_pilq_pkjq(registers &r) const;
    void fn_ijkl_piql_kpqj(registers &r) const;
    void fn_ijkl_piql_pkqj(registers &r) const;

};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_MUL_H
