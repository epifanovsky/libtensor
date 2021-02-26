#ifndef LIBTENSOR_TO_EWMULT2_IMPL_H
#define LIBTENSOR_TO_EWMULT2_IMPL_H

#include <memory>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_mul2.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../to_ewmult2_dims.h"
#include "../to_ewmult2.h"


namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
const char *to_ewmult2<N, M, K, T>::k_clazz = "to_ewmult2<N, M, K>";


template<size_t N, size_t M, size_t K, typename T>
to_ewmult2<N, M, K, T>::to_ewmult2(dense_tensor_rd_i<k_ordera, T> &ta,
    const tensor_transf<k_ordera, T> &tra,
    dense_tensor_rd_i<k_orderb, T> &tb,
    const tensor_transf<k_orderb, T> &trb,
    const tensor_transf<k_orderc, T> &trc) :

    m_ta(ta), m_perma(tra.get_perm()),
    m_tb(tb), m_permb(trb.get_perm()),
    m_permc(trc.get_perm()),
    m_d(tra.get_scalar_tr().get_coeff() * trb.get_scalar_tr().get_coeff() *
        trc.get_scalar_tr().get_coeff()),
    m_dimsc(to_ewmult2_dims<N, M, K>(ta.get_dims(), tra.get_perm(),
        tb.get_dims(), trb.get_perm(), trc.get_perm()).get_dimsc()) {

}


template<size_t N, size_t M, size_t K, typename T>
to_ewmult2<N, M, K, T>::to_ewmult2(dense_tensor_rd_i<k_ordera, T> &ta,
    dense_tensor_rd_i<k_orderb, T> &tb, T d) :

    m_ta(ta), m_tb(tb), m_d(d),
    m_dimsc(to_ewmult2_dims<N, M, K>(ta.get_dims(), permutation<k_ordera>(),
        tb.get_dims(), permutation<k_orderb>(), permutation<k_orderc>()).
        get_dimsc()) {

}


template<size_t N, size_t M, size_t K, typename T>
to_ewmult2<N, M, K, T>::to_ewmult2(dense_tensor_rd_i<k_ordera, T> &ta,
    const permutation<k_ordera> &perma, dense_tensor_rd_i<k_orderb, T> &tb,
    const permutation<k_orderb> &permb, const permutation<k_orderc> &permc,
    T d) :

    m_ta(ta), m_perma(perma), m_tb(tb), m_permb(permb), m_permc(permc),
    m_d(d),
    m_dimsc(to_ewmult2_dims<N, M, K>(ta.get_dims(), perma, tb.get_dims(), permb,
        permc).get_dimsc()) {

}


template<size_t N, size_t M, size_t K, typename T>
to_ewmult2<N, M, K, T>::~to_ewmult2() {

}


template<size_t N, size_t M, size_t K, typename T>
void to_ewmult2<N, M, K, T>::prefetch() {

    dense_tensor_rd_ctrl<k_ordera, T>(m_ta).req_prefetch();
    dense_tensor_rd_ctrl<k_orderb, T>(m_tb).req_prefetch();
}


template<size_t N, size_t M, size_t K, typename T>
void to_ewmult2<N, M, K, T>::perform(bool zero,
    dense_tensor_wr_i<k_orderc, T> &tc) {

    static const char *method = "perform(bool, "
        "dense_tensor_wr_i<k_orderc, T>&)";

    if(!m_dimsc.equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    to_ewmult2<N, M, K, T>::start_timer();

    try {

    dense_tensor_rd_ctrl<k_ordera, T> ca(m_ta);
    dense_tensor_rd_ctrl<k_orderb, T> cb(m_tb);
    dense_tensor_wr_ctrl<k_orderc, T> cc(tc);
    ca.req_prefetch();
    cb.req_prefetch();
    cc.req_prefetch();

    const dimensions<k_ordera> &dimsa = m_ta.get_dims();
    const dimensions<k_orderb> &dimsb = m_tb.get_dims();
    const dimensions<k_orderc> &dimsc = tc.get_dims();

    sequence<k_ordera, size_t> ma(0);
    sequence<k_orderb, size_t> mb(0);
    sequence<k_orderc, size_t> mc(0);
    for(size_t i = 0; i < k_ordera; i++) ma[i] = i;
    for(size_t i = 0; i < k_orderb; i++) mb[i] = i;
    for(size_t i = 0; i < k_orderc; i++) mc[i] = i;
    m_perma.apply(ma);
    m_permb.apply(mb);
    m_permc.apply(mc);

    std::list< loop_list_node<2, 1> > loop_in, loop_out;
    //  i runs over indexes in C
    //  m[i] runs over the "standard" index ordering
    for(size_t i = 0; i < k_orderc; i++) {
        typename std::list< loop_list_node<2, 1> >::iterator inode =
            loop_in.insert(loop_in.end(),
                loop_list_node<2, 1>(dimsc[i]));
        inode->stepb(0) = dimsc.get_increment(i);
        if(mc[i] < N) {
            size_t j = mc[i];
            inode->stepa(0) = dimsa.get_increment(ma[j]);
            inode->stepa(1) = 0;
        } else if(mc[i] < N + M) {
            size_t j = mc[i] - N;
            inode->stepa(0) = 0;
            inode->stepa(1) = dimsb.get_increment(mb[j]);
        } else {
            size_t j = mc[i] - N - M;
            inode->stepa(0) = dimsa.get_increment(ma[N + j]);
            inode->stepa(1) = dimsb.get_increment(mb[M + j]);
        }
    }

    const T *pa = ca.req_const_dataptr();
    const T *pb = cb.req_const_dataptr();
    T *pc = cc.req_dataptr();

    if(zero) {
        to_ewmult2<N, M, K, T>::start_timer("zero");
        size_t sz = dimsc.get_size();
        for(size_t i = 0; i < sz; i++) pc[i] = 0.0;
        to_ewmult2<N, M, K, T>::stop_timer("zero");
    }

    loop_registers_x<2, 1, T> r;
    r.m_ptra[0] = pa;
    r.m_ptra[1] = pb;
    r.m_ptrb[0] = pc;
    r.m_ptra_end[0] = pa + dimsa.get_size();
    r.m_ptra_end[1] = pb + dimsb.get_size();
    r.m_ptrb_end[0] = pc + dimsc.get_size();

    std::unique_ptr< kernel_base<linalg, 2, 1, T> > kern(
        kern_mul2<linalg, T>::match(m_d, loop_in, loop_out));
    to_ewmult2<N, M, K, T>::start_timer(kern->get_name());
    loop_list_runner_x<linalg, 2, 1, T>(loop_in).run(0, r, *kern);
    to_ewmult2<N, M, K, T>::stop_timer(kern->get_name());

    cc.ret_dataptr(pc); pc = 0;
    cb.ret_const_dataptr(pb); pb = 0;
    ca.ret_const_dataptr(pa); pa = 0;

    } catch (...) {
        to_ewmult2<N, M, K, T>::stop_timer();
        throw;
    }

    to_ewmult2<N, M, K, T>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TO_EWMULT2_IMPL_H
