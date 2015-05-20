#ifndef LIBTENSOR_TOD_DIAG_IMPL_H
#define LIBTENSOR_TOD_DIAG_IMPL_H

#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/kern_dadd1.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_diag.h"
#include "../tod_set.h"

namespace libtensor {


template<size_t N, size_t M>
const char *tod_diag<N, M>::k_clazz = "tod_diag<N, M>";


template<size_t N, size_t M>
tod_diag<N, M>::tod_diag(dense_tensor_rd_i<NA, double> &t,
    const sequence<NA, size_t> &m, const tensor_transf<NB, double> &tr) :

    m_t(t), m_mask(m), m_perm(tr.get_perm()),
    m_c(tr.get_scalar_tr().get_coeff()),
    m_dims(mk_dims(t.get_dims(), m_mask)) {

    m_dims.permute(m_perm);
}


template<size_t N, size_t M>
void tod_diag<N, M>::perform(bool zero, dense_tensor_wr_i<NB, double> &tb) {

    static const char *method =
            "perform(bool, dense_tensor_wr_i<M, double> &)";

#ifdef LIBTENSOR_DEBUG
    if(!tb.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }
#endif

    if(m_c == 0.0) {
        if (zero) {
            tod_diag<N, M>::start_timer("zero");
            tod_set<NB>().perform(zero, tb);
            tod_diag<N, M>::stop_timer("zero");
        }
        return;
    }

    tod_diag<N, M>::start_timer();

    try {
        dense_tensor_rd_ctrl<NA, double> ca(m_t);
        dense_tensor_wr_ctrl<NB, double> cb(tb);

        ca.req_prefetch();
        cb.req_prefetch();

        const dimensions<NA> &dimsa = m_t.get_dims();
        const dimensions<NB> &dimsb = tb.get_dims();

        //  Mapping of unpermuted indexes in b to permuted ones
        //
        sequence<NB, size_t> ib(0);
        for(size_t i = 0; i < NB; i++) ib[i] = i;
        permutation<NB> pinv(m_perm, true);
        pinv.apply(ib);

        std::list< loop_list_node<1, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<1, 1> >::iterator inode =
                loop_in.end();

        mask<NA> done;
        size_t iboffs = 0;

        for(size_t idxa = 0; idxa < NA; idxa++) {

            size_t inca = 0, incb = 0, len = 0;
            if(m_mask[idxa] != 0) {
                if(done[idxa]) {
                    iboffs++; continue;
                }

                size_t diag_idx = m_mask[idxa];

                //  Compute the stride on the diagonal
                //
                for(size_t j = idxa; j < NA; j++) {
                    if(m_mask[j] != diag_idx) continue;
                    inca += dimsa.get_increment(j);
                    done[j] = true;
                }
                incb = dimsb.get_increment(ib[idxa - iboffs]);
                len = dimsa.get_dim(idxa);
            } else {

                //  Compute the stride off the diagonal
                //  concatenating indexes if possible
                //
                len = 1;
                size_t idxb = ib[idxa - iboffs];
                while(idxa < N && m_mask[idxa] == 0 &&
                        idxb == ib[idxa - iboffs]) {

                    len *= dimsa.get_dim(idxa);
                    idxa++;
                    idxb++;
                }
                idxa--; idxb--;
                inca = dimsa.get_increment(idxa);
                incb = dimsb.get_increment(idxb);
            }

            inode = loop_in.insert(loop_in.end(), loop_list_node<1, 1>(len));
            inode->stepa(0) = inca;
            inode->stepb(0) = incb;
        }

        const double *pa = ca.req_const_dataptr();
        double *pb = cb.req_dataptr();

        loop_registers<1, 1> regs;
        regs.m_ptra[0] = pa;
        regs.m_ptrb[0] = pb;
        regs.m_ptra_end[0] = pa + dimsa.get_size();
        regs.m_ptrb_end[0] = pb + dimsb.get_size();

        {
            std::auto_ptr< kernel_base<linalg, 1, 1> > kern(
                    zero ? kern_dcopy<linalg>::match(m_c, loop_in, loop_out) :
                           kern_dadd1<linalg>::match(m_c, loop_in, loop_out));
            tod_diag<N, M>::start_timer(kern->get_name());
            loop_list_runner<linalg, 1, 1>(loop_in).run(0, regs, *kern);
            tod_diag<N, M>::start_timer(kern->get_name());
        }

        cb.ret_dataptr(pb); pb = 0;
        ca.ret_const_dataptr(pa); pa = 0;

    } catch(...) {
        tod_diag<N, M>::stop_timer();
        throw;
    }


    tod_diag<N, M>::stop_timer();
}


template<size_t N, size_t M>
dimensions<M> tod_diag<N, M>::mk_dims(const dimensions<N> &dims,
    const sequence<N, size_t> &msk) {

    static const char *method =
        "mk_dims(const dimensions<N> &, const mask<N>&)";

    //  Compute output dimensions
    //
    index<NB> i1, i2;

    bool bad_dims = false;
    sequence<NB + 1, size_t> d(0);
    size_t m = 0;
    for(size_t i = 0; i < NA; i++) {
        if (msk[i] != 0) {
            if (d[msk[i]] == 0) {
                d[msk[i]] = dims[i];
                i2[m++] = dims[i] - 1;
            }
            else {
                bad_dims = bad_dims || d[msk[i]] != dims[i];
            }
        }
        else {
            if (! bad_dims) i2[m++] = dims[i] - 1;
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

#endif // LIBTENSOR_TOD_DIAG_IMPL_H
