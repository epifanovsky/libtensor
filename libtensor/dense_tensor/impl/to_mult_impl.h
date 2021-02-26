#ifndef LIBTENSOR_TO_MULT_IMPL_H
#define LIBTENSOR_TO_MULT_IMPL_H

#include <memory>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_div2.h>
#include <libtensor/kernels/kern_mul2.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../to_mult.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_mult<N, T>::k_clazz = "to_mult<N, T>";


template<size_t N, typename T>
to_mult<N, T>::to_mult(
        dense_tensor_rd_i<N, T> &ta, const tensor_transf<N, T> &tra,
        dense_tensor_rd_i<N, T> &tb, const tensor_transf<N, T> &trb,
        bool recip, const scalar_transf<T> &trc) :

        m_ta(ta), m_tb(tb), m_perma(tra.get_perm()), m_permb(trb.get_perm()),
        m_recip(recip), m_c(trc.get_coeff()), m_dimsc(ta.get_dims()) {

    static const char *method = "to_mult("
            "dense_tensor_rd_i<N, T>&, const tensor_transf<N, T> &, "
            "dense_tensor_rd_i<N, T>&, const tensor_transf<N, T> &, "
            "bool, const scalar_transf<T> &)";

    m_dimsc.permute(m_perma);
    dimensions<N> dimsb(tb.get_dims());
    dimsb.permute(m_permb);

    if(!m_dimsc.equals(dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
                "ta, tb");
    }
    if (recip && trb.get_scalar_tr().get_coeff() == 0.0) {
        throw bad_parameter(g_ns, k_clazz, "to_mult()",
                __FILE__, __LINE__, "trb");
    }

    m_c = m_c * (recip ?
            tra.get_scalar_tr().get_coeff() / trb.get_scalar_tr().get_coeff() :
            tra.get_scalar_tr().get_coeff() * trb.get_scalar_tr().get_coeff());
}


template<size_t N, typename T>
to_mult<N, T>::to_mult(dense_tensor_rd_i<N, T> &ta,
    dense_tensor_rd_i<N, T> &tb, bool recip, T c) :

    m_ta(ta), m_tb(tb), m_recip(recip), m_c(c), m_dimsc(ta.get_dims()) {

    static const char *method = "to_mult(dense_tensor_rd_i<N, T>&, "
        "dense_tensor_rd_i<N, T>&, bool, T)";

    if(!ta.get_dims().equals(tb.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta,tb");
    }
}


template<size_t N, typename T>
to_mult<N, T>::to_mult(
        dense_tensor_rd_i<N, T> &ta, const permutation<N> &pa,
        dense_tensor_rd_i<N, T> &tb, const permutation<N> &pb,
        bool recip, T c) :

        m_ta(ta), m_tb(tb), m_perma(pa), m_permb(pb),
        m_recip(recip), m_c(c), m_dimsc(ta.get_dims()) {

    static const char *method = "to_mult("
            "dense_tensor_rd_i<N, T>&, permutation<N>, "
            "dense_tensor_rd_i<N, T>&, permutation<N>, "
            "bool, T)";

    m_dimsc.permute(pa);
    dimensions<N> dimsb(tb.get_dims());
    dimsb.permute(pb);

    if(!m_dimsc.equals(dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
                "ta, tb");
    }
}


template<size_t N, typename T>
void to_mult<N, T>::prefetch() {

    dense_tensor_rd_ctrl<N, T>(m_ta).req_prefetch();
    dense_tensor_rd_ctrl<N, T>(m_tb).req_prefetch();

}


template<size_t N, typename T>
void to_mult<N, T>::perform(bool zero, dense_tensor_wr_i<N, T> &tc) {

    static const char *method = "perform(bool, T, "
        "dense_tensor_wr_i<N, T>&)";

    if(!m_dimsc.equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    to_mult<N, T>::start_timer();

    try {

    dense_tensor_rd_ctrl<N, T> ca(m_ta), cb(m_tb);
    dense_tensor_wr_ctrl<N, T> cc(tc);
    ca.req_prefetch();
    cb.req_prefetch();
    cc.req_prefetch();

    const dimensions<N> &dimsa = m_ta.get_dims();
    const dimensions<N> &dimsb = m_tb.get_dims();
    const dimensions<N> &dimsc = tc.get_dims();

    sequence<N, size_t> mapa(0), mapb(0);
    for(size_t i = 0; i < N; i++) mapa[i] = mapb[i] = i;
    m_perma.apply(mapa);
    m_permb.apply(mapb);

    std::list< loop_list_node<2, 1> > loop_in, loop_out;
    typename std::list< loop_list_node<2, 1> >::iterator inode = loop_in.end();
    for (size_t idxc = 0; idxc < N; ) {
        size_t len = 1;
        size_t idxa = mapa[idxc], idxb = mapb[idxc];

        do {
            len *= dimsa.get_dim(idxa);
            idxa++; idxb++; idxc++;
        } while (idxc < N && mapa[idxc] == idxa && mapb[idxc] == idxb);

        inode = loop_in.insert(loop_in.end(), loop_list_node<2, 1>(len));
        inode->stepa(0) = dimsa.get_increment(idxa - 1);
        inode->stepa(1) = dimsb.get_increment(idxb - 1);
        inode->stepb(0) = dimsc.get_increment(idxc - 1);
    }

    const T *pa = ca.req_const_dataptr();
    const T *pb = cb.req_const_dataptr();
    T *pc = cc.req_dataptr();

    if(zero) {
        to_mult<N, T>::start_timer("zero");
        size_t sz = dimsc.get_size();
        for(size_t i = 0; i < sz; i++) pc[i] = 0.0;
        to_mult<N, T>::stop_timer("zero");
    }

    loop_registers_x<2, 1, T> r;
    r.m_ptra[0] = pa;
    r.m_ptra[1] = pb;
    r.m_ptrb[0] = pc;
    r.m_ptra_end[0] = pa + dimsa.get_size();
    r.m_ptra_end[1] = pb + dimsb.get_size();
    r.m_ptrb_end[0] = pc + dimsc.get_size();

    std::unique_ptr< kernel_base<linalg, 2, 1, T> > kern(
        m_recip ?
            kern_div2<T>::match(m_c, loop_in, loop_out) :
            kern_mul2<linalg, T>::match(m_c, loop_in, loop_out));
    to_mult<N, T>::start_timer(kern->get_name());
    loop_list_runner_x<linalg, 2, 1, T>(loop_in).run(0, r, *kern);
    to_mult<N, T>::stop_timer(kern->get_name());

    cc.ret_dataptr(pc); pc = 0;
    cb.ret_const_dataptr(pb); pb = 0;
    ca.ret_const_dataptr(pa); pa = 0;

    } catch (...) {
        to_mult<N, T>::stop_timer();
        throw;
    }

    to_mult<N, T>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TO_MULT_IMPL_H
