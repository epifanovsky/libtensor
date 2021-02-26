#ifndef LIBTENSOR_TOD_EXTRACT_IMPL_H
#define LIBTENSOR_TOD_EXTRACT_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/kernels/kern_dadd1.h>
#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_extract.h"

namespace libtensor {


template<size_t N, size_t M>
const char *tod_extract<N, M>::k_clazz = "tod_extract<N, M>";


template<size_t N, size_t M>
tod_extract<N, M>::tod_extract(
        dense_tensor_rd_i<NA, double> &t, const mask<NA> &m,
        const index<NA> &idx, const tensor_transf_type &tr) :

    m_t(t), m_mask(m), m_perm(tr.get_perm()),
    m_c(tr.get_scalar_tr().get_coeff()),
    m_dims(mk_dims(t.get_dims(), m_mask)), m_idx(idx) {

    m_dims.permute(m_perm);
}


template<size_t N, size_t M>
tod_extract<N, M>::tod_extract(
        dense_tensor_rd_i<NA, double> &t, const mask<NA> &m,
        const index<NA> &idx, double c) :

    m_t(t), m_mask(m), m_c(c), m_dims(mk_dims(t.get_dims(), m)), m_idx(idx) {

}


template<size_t N, size_t M>
tod_extract<N, M>::tod_extract(
        dense_tensor_rd_i<NA, double> &t, const mask<NA> &m,
        const index<NA> &idx, const permutation<NB> &p, double c) :

    m_t(t), m_mask(m), m_perm(p), m_c(c),
    m_dims(mk_dims(t.get_dims(), m)), m_idx(idx) {

    m_dims.permute(p);
}


template<size_t N, size_t M>
void tod_extract<N, M>::perform(bool zero, dense_tensor_wr_i<NB, double> &tb) {

    static const char *method =
            "perform(bool, dense_tensor_wr_i<N - M, double>&)";

    if(!tb.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    tod_extract<N, M>::start_timer();

    try {

        dense_tensor_rd_ctrl<NA, double> ca(m_t);
        dense_tensor_wr_ctrl<NB, double> cb(tb);

        ca.req_prefetch();
        cb.req_prefetch();

        const dimensions<NA> &dimsa = m_t.get_dims();
        const dimensions<NB> &dimsb = tb.get_dims();

        //  Mapping of unpermuted indexes in b to permuted ones
        //
        sequence<NB, size_t> seqb(0);
        for(size_t i = 0; i < NB; i++) seqb[i] = i;
        m_perm.apply(seqb);

        std::list< loop_list_node<1, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<1, 1> >::iterator inode =
                loop_in.end();

        size_t iboffs = 0;
        for(size_t idxa = 0; idxa < N; idxa++) {

            size_t len = 0;
            if(! m_mask[idxa]) {
                iboffs++; continue;
            }

            // situation if the index is not fixed
            len = 1;
            size_t idxb = seqb[idxa - iboffs];
            while(idxa < NA && m_mask[idxa] && idxb == seqb[idxa - iboffs]) {
                len *= dimsa.get_dim(idxa);
                idxa++;
                idxb++;
            }
            idxa--; idxb--;

            inode = loop_in.insert(loop_in.end(), loop_list_node<1, 1>(len));
            inode->stepa(0) = dimsa.get_increment(idxa);
            inode->stepb(0) = dimsb.get_increment(idxb);
        }

        const double *pa = ca.req_const_dataptr();
        double *pb = cb.req_dataptr();

        size_t pa_offset = 0;
        for(size_t i = 0; i < NA; i++) {
            if(m_idx[i] != 0) {
                pa_offset += m_idx[i] * dimsa.get_increment(i);
            }
        }

        loop_registers<1, 1> regs;
        regs.m_ptra[0] = pa + pa_offset;
        regs.m_ptrb[0] = pb;
        regs.m_ptra_end[0] = pa + dimsa.get_size();
        regs.m_ptrb_end[0] = pb + dimsb.get_size();

        {
            std::unique_ptr< kernel_base<linalg, 1, 1, double> > kern(
                zero ?
                    kern_dcopy<linalg>::match(m_c, loop_in, loop_out) :
                    kern_dadd1<linalg>::match(m_c, loop_in, loop_out));
            tod_extract<N, M>::start_timer(kern->get_name());
            loop_list_runner<linalg, 1, 1>(loop_in).run(0, regs, *kern);
            tod_extract<N, M>::stop_timer(kern->get_name());
        }

        cb.ret_dataptr(pb); pb = 0;
        ca.ret_const_dataptr(pa); pa = 0;

    } catch(...) {
        tod_extract<N, M>::stop_timer();
        throw;
    }

    tod_extract<N, M>::stop_timer();
}


template<size_t N, size_t M>
dimensions<N - M> tod_extract<N, M>::mk_dims(const dimensions<NA> &dims,
    const mask<NA> &msk) {

    static const char *method = "mk_dims(const dimensions<N> &, "
        "const mask<N>&)";

    //  Compute output dimensions
    //
    index<NB> i1, i2;

    size_t m = 0, j = 0;
    bool bad_dims = false;
    for(size_t i = 0; i < N; i++) {
        if(msk[i]){
            i2[j++] = dims[i] - 1;
        }else{
            m++;
        }
    }
    if(m != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "m");
    }
    if(bad_dims) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }
    return dimensions<NB>(index_range<NB>(i1, i2));
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_EXTRACT_IMPL_H
