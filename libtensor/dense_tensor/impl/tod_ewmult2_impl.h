#ifndef LIBTENSOR_TOD_EWMULT2_IMPL_H
#define LIBTENSOR_TOD_EWMULT2_IMPL_H

#include <memory>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dmul2.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_ewmult2.h"


namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *tod_ewmult2<N, M, K>::k_clazz = "tod_ewmult2<N, M, K>";


template<size_t N, size_t M, size_t K>
tod_ewmult2<N, M, K>::tod_ewmult2(dense_tensor_rd_i<k_ordera, double> &ta,
    const tensor_transf<k_ordera, double> &tra,
    dense_tensor_rd_i<k_orderb, double> &tb,
    const tensor_transf<k_orderb, double> &trb,
    const tensor_transf<k_orderc, double> &trc) :

    m_ta(ta), m_perma(tra.get_perm()),
    m_tb(tb), m_permb(trb.get_perm()),
    m_permc(trc.get_perm()), m_d(trc.get_scalar_tr().get_coeff()),
    m_dimsc(make_dimsc(ta.get_dims(), tra.get_perm(),
            tb.get_dims(), trb.get_perm(), trc.get_perm())) {

}


template<size_t N, size_t M, size_t K>
tod_ewmult2<N, M, K>::tod_ewmult2(dense_tensor_rd_i<k_ordera, double> &ta,
    dense_tensor_rd_i<k_orderb, double> &tb, double d) :

    m_ta(ta), m_tb(tb), m_d(d),
    m_dimsc(make_dimsc(ta.get_dims(), permutation<k_ordera>(),
        tb.get_dims(), permutation<k_orderb>(),
        permutation<k_orderc>())) {

}


template<size_t N, size_t M, size_t K>
tod_ewmult2<N, M, K>::tod_ewmult2(dense_tensor_rd_i<k_ordera, double> &ta,
    const permutation<k_ordera> &perma, dense_tensor_rd_i<k_orderb, double> &tb,
    const permutation<k_orderb> &permb, const permutation<k_orderc> &permc,
    double d) :

    m_ta(ta), m_perma(perma), m_tb(tb), m_permb(permb), m_permc(permc),
    m_d(d),
    m_dimsc(make_dimsc(ta.get_dims(), perma, tb.get_dims(), permb, permc)) {

}


template<size_t N, size_t M, size_t K>
tod_ewmult2<N, M, K>::~tod_ewmult2() {

}


template<size_t N, size_t M, size_t K>
void tod_ewmult2<N, M, K>::prefetch() {

    dense_tensor_rd_ctrl<k_ordera, double>(m_ta).req_prefetch();
    dense_tensor_rd_ctrl<k_orderb, double>(m_tb).req_prefetch();
}


template<size_t N, size_t M, size_t K>
void tod_ewmult2<N, M, K>::perform(bool zero,
    dense_tensor_wr_i<k_orderc, double> &tc) {

    static const char *method = "perform(bool, "
        "dense_tensor_wr_i<k_orderc, double>&)";

    if(!m_dimsc.equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    tod_ewmult2<N, M, K>::start_timer();

    try {

    dense_tensor_rd_ctrl<k_ordera, double> ca(m_ta);
    dense_tensor_rd_ctrl<k_orderb, double> cb(m_tb);
    dense_tensor_wr_ctrl<k_orderc, double> cc(tc);
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

    const double *pa = ca.req_const_dataptr();
    const double *pb = cb.req_const_dataptr();
    double *pc = cc.req_dataptr();

    if(zero) {
        tod_ewmult2<N, M, K>::start_timer("zero");
        size_t sz = dimsc.get_size();
        for(size_t i = 0; i < sz; i++) pc[i] = 0.0;
        tod_ewmult2<N, M, K>::stop_timer("zero");
    }

    loop_registers<2, 1> r;
    r.m_ptra[0] = pa;
    r.m_ptra[1] = pb;
    r.m_ptrb[0] = pc;
    r.m_ptra_end[0] = pa + dimsa.get_size();
    r.m_ptra_end[1] = pb + dimsb.get_size();
    r.m_ptrb_end[0] = pc + dimsc.get_size();

    std::auto_ptr< kernel_base<linalg, 2, 1> > kern(
        kern_dmul2<linalg>::match(m_d, loop_in, loop_out));
    tod_ewmult2<N, M, K>::start_timer(kern->get_name());
    loop_list_runner<linalg, 2, 1>(loop_in).run(0, r, *kern);
    tod_ewmult2<N, M, K>::stop_timer(kern->get_name());

    cc.ret_dataptr(pc); pc = 0;
    cb.ret_const_dataptr(pb); pb = 0;
    ca.ret_const_dataptr(pa); pa = 0;

    } catch (...) {
        tod_ewmult2<N, M, K>::stop_timer();
        throw;
    }

    tod_ewmult2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
dimensions<N + M + K> tod_ewmult2<N, M, K>::make_dimsc(
    const dimensions<k_ordera> &dimsa, const permutation<k_ordera> &perma,
    const dimensions<k_orderb> &dimsb, const permutation<k_orderb> &permb,
    const permutation<k_orderc> &permc) {

    static const char *method = "make_dimsc()";

    dimensions<k_ordera> dimsa1(dimsa);
    dimsa1.permute(perma);
    dimensions<k_orderb> dimsb1(dimsb);
    dimsb1.permute(permb);

    index<k_orderc> i1, i2;
    for(size_t i = 0; i != N; i++) i2[i] = dimsa1[i] - 1;
    for(size_t i = 0; i != M; i++) i2[N + i] = dimsb1[i] - 1;
    for(size_t i = 0; i != K; i++) {
        if(dimsa1[N + i] != dimsb1[M + i]) {
            throw bad_dimensions(g_ns, k_clazz, method,
                __FILE__, __LINE__, "ta,tb");
        }
        i2[N + M + i] = dimsa1[N + i] - 1;
    }
    dimensions<k_orderc> dimsc(index_range<k_orderc>(i1, i2));
    dimsc.permute(permc);
    return dimsc;
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_EWMULT2_IMPL_H
