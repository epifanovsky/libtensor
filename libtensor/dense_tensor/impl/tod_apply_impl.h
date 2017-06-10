#ifndef LIBTENSOR_TOD_APPLY_IMPL_H
#define LIBTENSOR_TOD_APPLY_IMPL_H

#include <libtensor/kernels/kern_apply.h>
#include <libtensor/kernels/kern_applyadd.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_apply.h"

namespace libtensor {


template<size_t N, typename Functor>
const char *tod_apply<N, Functor>::k_clazz = "tod_apply<N, Functor>";


template<size_t N, typename Functor>
tod_apply<N, Functor>::tod_apply(
        dense_tensor_rd_i<N, double> &ta,
        const Functor &fn,
        const scalar_transf_type &tr1,
        const tensor_transf_type &tr2) :

    m_ta(ta), m_fn(fn), m_c1(tr1.get_coeff()),
    m_c2(tr2.get_scalar_tr().get_coeff()), m_permb(tr2.get_perm()),
    m_dimsb(mk_dimsb(m_ta, tr2.get_perm())) {

}


template<size_t N, typename Functor>
tod_apply<N, Functor>::tod_apply(dense_tensor_rd_i<N, double> &ta,
    const Functor &fn, double c) :

    m_ta(ta), m_fn(fn), m_c1(c), m_c2(1.0), m_dimsb(m_ta.get_dims()) {

}


template<size_t N, typename Functor>
tod_apply<N, Functor>::tod_apply(dense_tensor_rd_i<N, double> &ta,
    const Functor &fn, const permutation<N> &p, double c) :

    m_ta(ta), m_fn(fn), m_c1(c), m_c2(1.0), m_permb(p),
    m_dimsb(mk_dimsb(ta, p)) {

}


template<size_t N, typename Functor>
void tod_apply<N, Functor>::perform(
        bool zero, dense_tensor_wr_i<N, double> &tb) {

    static const char *method =
        "perform(bool, dense_tensor_wr_i<N, double>&)";

    if(!tb.get_dims().equals(m_dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }
    if(! zero && m_c2 == 0.0) return;

    tod_apply<N, Functor>::start_timer();

    try {

    dense_tensor_rd_ctrl<N, double> ca(m_ta);
    dense_tensor_wr_ctrl<N, double> cb(tb);
    ca.req_prefetch();
    cb.req_prefetch();

    const dimensions<N> &dimsa = m_ta.get_dims();
    const dimensions<N> &dimsb = tb.get_dims();

    sequence<N, size_t> seqa;
    for(register size_t i = 0; i < N; i++) seqa[i] = i;
    m_permb.apply(seqa);


    std::list< loop_list_node<1, 1> > loop_in, loop_out;
    typename std::list< loop_list_node<1, 1> >::iterator inode =
            loop_in.end();

    for(size_t idxb = 0; idxb < N;) {
        size_t len = 1;
        size_t idxa = seqa[idxb];
        do {
            len *= dimsa.get_dim(idxa);
            idxa++; idxb++;
        } while(idxb < N && seqa[idxb] == idxa);

        inode = loop_in.insert(loop_in.end(), loop_list_node<1, 1>(len));
        inode->stepa(0) = dimsa.get_increment(idxa - 1);
        inode->stepb(0) = dimsb.get_increment(idxb - 1);
    }

    const double *pa = ca.req_const_dataptr();
    double *pb = cb.req_dataptr();

    loop_registers<1, 1> r;
    r.m_ptra[0] = pa;
    r.m_ptrb[0] = pb;
    r.m_ptra_end[0] = pa + dimsa.get_size();
    r.m_ptrb_end[0] = pb + dimsb.get_size();

    {
        std::auto_ptr< kernel_base<linalg, 1, 1, double> > kern(zero ?
            kern_apply<Functor>::match(m_fn, m_c1, m_c2, loop_in, loop_out) :
            kern_applyadd<Functor>::match(m_fn, m_c1, m_c2, loop_in, loop_out));
        tod_apply<N, Functor>::start_timer(kern->get_name());
        loop_list_runner<linalg, 1, 1>(loop_in).run(0, r, *kern);
        tod_apply<N, Functor>::stop_timer(kern->get_name());
    }

    ca.ret_const_dataptr(pa);
    cb.ret_dataptr(pb);

    } catch(...) {
        tod_apply<N, Functor>::stop_timer();
        throw;
    }
    tod_apply<N, Functor>::stop_timer();
}


template<size_t N, typename Functor>
dimensions<N> tod_apply<N, Functor>::mk_dimsb(dense_tensor_rd_i<N, double> &ta,
    const permutation<N> &perm) {

    dimensions<N> dims(ta.get_dims());
    dims.permute(perm);
    return dims;
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_APPLY_IMPL_H
