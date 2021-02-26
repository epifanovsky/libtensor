#ifndef LIBTENSOR_TO_DIAG_IMPL_H
#define LIBTENSOR_TO_DIAG_IMPL_H

#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/kern_dadd1.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../to_diag_dims.h"
#include "../to_diag.h"
#include "../to_set.h"

namespace libtensor {


template<size_t N, size_t M, typename T>
const char *to_diag<N, M, T>::k_clazz = "to_diag<N, M, T>";


template<size_t N, size_t M, typename T>
to_diag<N, M, T>::to_diag(dense_tensor_rd_i<NA, T> &t,
    const sequence<NA, size_t> &m, const tensor_transf<NB, T> &tr) :

    m_t(t), m_mask(m), m_perm(tr.get_perm()),
    m_c(tr.get_scalar_tr().get_coeff()),
    m_dims(to_diag_dims<N, M>(m_t.get_dims(), m_mask, m_perm).get_dimsb()) {

}


template<size_t N, size_t M, typename T>
void to_diag<N, M, T>::perform(bool zero, dense_tensor_wr_i<NB, T> &tb) {

    static const char *method =
            "perform(bool, dense_tensor_wr_i<M, T> &)";

#ifdef LIBTENSOR_DEBUG
    if(!tb.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }
#endif

    if(m_c == 0.0) {
        if (zero) {
            to_diag<N, M, T>::start_timer("zero");
            to_set<NB, T>().perform(zero, tb);
            to_diag<N, M, T>::stop_timer("zero");
        }
        return;
    }

    to_diag<N, M, T>::start_timer();

    try {
        dense_tensor_rd_ctrl<NA, T> ca(m_t);
        dense_tensor_wr_ctrl<NB, T> cb(tb);

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

        const T *pa = ca.req_const_dataptr();
        T *pb = cb.req_dataptr();

        loop_registers_x<1, 1, T> regs;
        regs.m_ptra[0] = pa;
        regs.m_ptrb[0] = pb;
        regs.m_ptra_end[0] = pa + dimsa.get_size();
        regs.m_ptrb_end[0] = pb + dimsb.get_size();

        {
            std::unique_ptr< kernel_base<linalg, 1, 1, T> > kern(
                    zero ? kern_copy<linalg, T>::match(m_c, loop_in, loop_out) :
                           kern_add1<linalg, T>::match(m_c, loop_in, loop_out));
            to_diag<N, M, T>::start_timer(kern->get_name());
            loop_list_runner_x<linalg, 1, 1, T>(loop_in).run(0, regs, *kern);
            to_diag<N, M, T>::start_timer(kern->get_name());
        }

        cb.ret_dataptr(pb); pb = 0;
        ca.ret_const_dataptr(pa); pa = 0;

    } catch(...) {
        to_diag<N, M, T>::stop_timer();
        throw;
    }


    to_diag<N, M, T>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TO_DIAG_IMPL_H
