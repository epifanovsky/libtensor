#ifndef LIBTENSOR_TOD_COPY_IMPL_H
#define LIBTENSOR_TOD_COPY_IMPL_H

#include <memory>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dadd1.h>
#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_set.h"
#include "../tod_copy.h"


namespace libtensor {


template<size_t N>
const char *tod_copy<N>::k_clazz = "tod_copy<N>";


template<size_t N>
tod_copy<N>::tod_copy(dense_tensor_rd_i<N, double> &ta, double c) :

    m_ta(ta), m_c(c), m_dimsb(mk_dimsb(m_ta, m_perm)) {

}


template<size_t N>
tod_copy<N>::tod_copy(dense_tensor_rd_i<N, double> &ta, const permutation<N> &p,
    double c) :

    m_ta(ta), m_perm(p), m_c(c), m_dimsb(mk_dimsb(ta, p)) {

}


template<size_t N>
tod_copy<N>::tod_copy(dense_tensor_rd_i<N, double> &ta,
        const tensor_transf<N, double> &tr) :

    m_ta(ta), m_perm(tr.get_perm()), m_c(tr.get_scalar_tr().get_coeff()),
    m_dimsb(mk_dimsb(ta, tr.get_perm())) {

}


template<size_t N>
void tod_copy<N>::prefetch() {

    dense_tensor_rd_ctrl<N, double>(m_ta).req_prefetch();
}


template<size_t N>
void tod_copy<N>::perform(bool zero, dense_tensor_wr_i<N, double> &tb) {

    static const char *method = "perform(bool, dense_tensor_wr_i<N, double>&)";

    if(!tb.get_dims().equals(m_dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    //  Special case
    if(m_c == 0.0) {
        if(zero) {
            tod_copy<N>::start_timer("zero");
            tod_set<N>().perform(zero, tb);
            tod_copy<N>::stop_timer("zero");
        }
        return;
    }

    tod_copy<N>::start_timer();

    try {

        dense_tensor_rd_ctrl<N, double> ca(m_ta);
        dense_tensor_wr_ctrl<N, double> cb(tb);

        ca.req_prefetch();
        cb.req_prefetch();

        const dimensions<N> &dimsa = m_ta.get_dims();
        const dimensions<N> &dimsb = tb.get_dims();

        sequence<N, size_t> seqa(0);
        for(register size_t i = 0; i < N; i++) seqa[i] = i;
        m_perm.apply(seqa);

        std::list< loop_list_node<1, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<1, 1> >::iterator inode =
            loop_in.end();

        //  Go over indexes in B and connect them with indexes in A
        //  trying to glue together consecutive indexes
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

        //  Invoke the appropriate kernel

        loop_registers<1, 1> r;
        r.m_ptra[0] = pa;
        r.m_ptrb[0] = pb;
        r.m_ptra_end[0] = pa + dimsa.get_size();
        r.m_ptrb_end[0] = pb + dimsb.get_size();

        {
            std::auto_ptr< kernel_base<linalg, 1, 1> > kern(
                zero ?
                    kern_dcopy<linalg>::match(m_c, loop_in, loop_out) :
                    kern_dadd1<linalg>::match(m_c, loop_in, loop_out));
            tod_copy<N>::start_timer(kern->get_name());
            loop_list_runner<linalg, 1, 1>(loop_in).run(0, r, *kern);
            tod_copy<N>::stop_timer(kern->get_name());
        }

        ca.ret_const_dataptr(pa);
        cb.ret_dataptr(pb);

    } catch(...) {
        tod_copy<N>::stop_timer();
        throw;
    }

    tod_copy<N>::stop_timer();
}


template<size_t N>
dimensions<N> tod_copy<N>::mk_dimsb(dense_tensor_rd_i<N, double> &ta,
    const permutation<N> &perm) {

    dimensions<N> dims(ta.get_dims());
    dims.permute(perm);
    return dims;
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_COPY_IMPL_H
