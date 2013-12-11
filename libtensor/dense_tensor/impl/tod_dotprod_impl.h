#ifndef LIBTENSOR_TOD_DOTPROD_IMPL_H
#define LIBTENSOR_TOD_DOTPROD_IMPL_H

#include <memory>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dmul2.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_dotprod.h"

namespace libtensor {


template<size_t N>
const char tod_dotprod<N>::k_clazz[] = "tod_dotprod<N>";


template<size_t N>
tod_dotprod<N>::tod_dotprod(dense_tensor_rd_i<N, double> &ta,
    dense_tensor_rd_i<N, double> &tb) :

    m_ta(ta), m_tb(tb), m_c(1.0) {

    static const char method[] = "tod_dotprod(dense_tensor_rd_i<N, double>&, "
        "dense_tensor_rd_i<N, double>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta != tb");
    }
}


template<size_t N>
tod_dotprod<N>::tod_dotprod(dense_tensor_rd_i<N, double> &ta,
    const permutation<N> &perma, dense_tensor_rd_i<N, double> &tb,
    const permutation<N> &permb) :

    m_ta(ta), m_tb(tb), m_perma(perma), m_permb(permb), m_c(1.0) {

    static const char method[] = "tod_dotprod(dense_tensor_rd_i<N, double>&, "
        "const permutation<N>&, dense_tensor_rd_i<N, double>&, "
        "const permutation<N>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta != tb");
    }
}


template<size_t N>
tod_dotprod<N>::tod_dotprod(
        dense_tensor_rd_i<N, double> &ta,
        const tensor_transf<N, double> &tra,
        dense_tensor_rd_i<N, double> &tb,
        const tensor_transf<N, double> &trb) :

    m_ta(ta), m_tb(tb), m_perma(tra.get_perm()), m_permb(trb.get_perm()),
    m_c(tra.get_scalar_tr().get_coeff() * trb.get_scalar_tr().get_coeff()){

    static const char method[] = "tod_dotprod(dense_tensor_rd_i<N, double>&, "
        "const tensor_transf<N, double>&, dense_tensor_rd_i<N, double>&, "
        "const tensor_transf<N, double>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta != tb");
    }
}


template<size_t N>
void tod_dotprod<N>::prefetch() {

    dense_tensor_rd_ctrl<N, double>(m_ta).req_prefetch();
    dense_tensor_rd_ctrl<N, double>(m_tb).req_prefetch();
}


template<size_t N>
double tod_dotprod<N>::calculate() {

    double result = 0.0;

    tod_dotprod<N>::start_timer();

    try {

        dense_tensor_rd_ctrl<N, double> ca(m_ta), cb(m_tb);
        ca.req_prefetch();
        cb.req_prefetch();

        permutation<N> pinvb(m_permb, true);

        sequence<N, size_t> seqa(0);
        for(size_t i = 0; i < N; i++) seqa[i] = i;
        m_perma.apply(seqa);
        pinvb.apply(seqa);

        const dimensions<N> &dimsa(m_ta.get_dims());
        const dimensions<N> &dimsb(m_tb.get_dims());

        std::list< loop_list_node<2, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<2, 1> >::iterator inode =
            loop_in.end();
        for(size_t ib = 0; ib < N;) {
            size_t len = 1;
            size_t ia = seqa[ib];
            do {
                len *= dimsa.get_dim(ia);
                ia++; ib++;
            } while(ib < N && seqa[ib] == ia);

            inode = loop_in.insert(loop_in.end(), loop_list_node<2, 1>(len));
            inode->stepa(0) = dimsa.get_increment(ia - 1);
            inode->stepa(1) = dimsb.get_increment(ib - 1);
            inode->stepb(0) = 0;
        }


        const double *pa = ca.req_const_dataptr();
        const double *pb = cb.req_const_dataptr();

        loop_registers<2, 1> r;
        r.m_ptra[0] = pa;
        r.m_ptra[1] = pb;
        r.m_ptrb[0] = &result;
        r.m_ptra_end[0] = pa + dimsa.get_size();
        r.m_ptra_end[1] = pb + dimsb.get_size();
        r.m_ptrb_end[0] = &result + 1;

        std::auto_ptr< kernel_base<linalg, 2, 1> > kern(
            kern_dmul2<linalg>::match(1.0, loop_in, loop_out));
        tod_dotprod<N>::start_timer(kern->get_name());
        loop_list_runner<linalg, 2, 1>(loop_in).run(0, r, *kern);
        tod_dotprod<N>::stop_timer(kern->get_name());

        ca.ret_const_dataptr(pa);
        cb.ret_const_dataptr(pb);

        result *= m_c;

    } catch(...) {
        tod_dotprod<N>::stop_timer();
        throw;
    }

    tod_dotprod<N>::stop_timer();

    return result;
}


template<size_t N>
bool tod_dotprod<N>::verify_dims() {

    dimensions<N> dimsa(m_ta.get_dims());
    dimensions<N> dimsb(m_tb.get_dims());
    dimsa.permute(m_perma);
    dimsb.permute(m_permb);
    return dimsa.equals(dimsb);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_IMPL_H
