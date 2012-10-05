#ifndef LIBTENSOR_TOD_CONV_DIAG_TENSOR_IMPL_H
#define LIBTENSOR_TOD_CONV_DIAG_TENSOR_IMPL_H

#include <list>
#include <memory>
#include <vector>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dadd1.h>
#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/tod/bad_dimensions.h>
#include "../diag_tensor_ctrl.h"
#include "../tod_conv_diag_tensor.h"

namespace libtensor {


template<size_t N>
const char *tod_conv_diag_tensor<N>::k_clazz = "tod_conv_diag_tensor<N>";


template<size_t N>
void tod_conv_diag_tensor<N>::perform(dense_tensor_wr_i<N, double> &tb) {

    static const char *method = "perform(dense_tensor_wr_i<N, double>&)";

    if(!m_ta.get_dims().equals(tb.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    perform(tb, index<N>());
}


template<size_t N>
void tod_conv_diag_tensor<N>::perform(
    dense_tensor_wr_i<N, double> &tb,
    const index<N> &off) {

    static const char *method = "perform(dense_tensor_wr_i<N, double>&, "
        "const index<N>&)";

    const dimensions<N> &dimsa = m_ta.get_dims();
    const dimensions<N> &dimsb = tb.get_dims();

    for(size_t i = 0; i < N; i++) {
        if(off[i] + dimsa[i] > dimsb[i]) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "off");
        }
    }

    //  Absolute offset in output tensor
    size_t aoff = 0;
    for(size_t i = 0; i < N; i++) aoff += off[i] * dimsb.get_increment(i);

    const diag_tensor_space<N> &dtsa = m_ta.get_space();
    diag_tensor_rd_ctrl<N, double> ca(m_ta);
    dense_tensor_wr_ctrl<N, double> cb(tb);

    double *pb = cb.req_dataptr();

    {
        //  Zero output window

        std::list< loop_list_node<1, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<1, 1> >::iterator inode =
            loop_in.end();
        for(size_t i = 0; i < N; i++) {
            inode = loop_in.insert(loop_in.end(),
                loop_list_node<1, 1>(dimsa[i]));
            inode->stepa(0) = 0;
            inode->stepb(0) = dimsb.get_increment(i);
        }

        double zero = 0.0;
        loop_registers<1, 1> r;
        r.m_ptra[0] = &zero;
        r.m_ptrb[0] = pb + aoff;
        r.m_ptra_end[0] = &zero + 1;
        r.m_ptrb_end[0] = pb + dimsb.get_size();

        {
            std::auto_ptr< kernel_base<linalg, 1, 1> >kern(
                kern_dcopy<linalg>::match(1.0, loop_in, loop_out));
            loop_list_runner<linalg, 1, 1>(loop_in).run(0, r, *kern);
        }
    }

    std::vector<size_t> ssl; // List of subspaces
    dtsa.get_all_subspaces(ssl);
    for(size_t ssi = 0; ssi < ssl.size(); ssi++) {

        size_t ssn = ssl[ssi];

        const diag_tensor_subspace<N> &ss = dtsa.get_subspace(ssn);

        std::list< loop_list_node<1, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<1, 1> >::iterator inode =
            loop_in.end();

        sequence<N, size_t> diags(N); // Diagonal numbers, N for unrestricted
        for(size_t idiag = 0; idiag < ss.get_ndiag(); idiag++) {
            const mask<N> &md = ss.get_diag_mask(idiag);
            for(size_t i = 0; i < N; i++) if(md[i]) diags[i] = idiag;
        }
        mask<N> mdone;
        for(size_t i = 0; i < N; i++) {

            if(mdone[i]) continue;

            size_t stepb = 0;
            if(diags[i] < N) {
                const mask<N> &m = ss.get_diag_mask(diags[i]);
                for(size_t j = 0; j < N; j++) {
                    if(m[j]) stepb += dimsb.get_increment(j);
                }
                mdone |= m;
            } else {
                stepb = dimsb.get_increment(i);
                mdone[i] = true;
            }
            size_t w = dimsa[i];
            for(typename std::list< loop_list_node<1, 1> >::iterator jnode =
                loop_in.begin(); jnode != loop_in.end(); ++jnode) {
                jnode->stepa(0) *= w;
            }
            inode = loop_in.insert(loop_in.end(), loop_list_node<1, 1>(w));
            inode->stepa(0) = 1;
            inode->stepb(0) = stepb;
        }
#ifdef LIBTENSOR_DEBUG
        if(loop_in.begin()->stepa(0) * loop_in.begin()->weight() !=
            dtsa.get_subspace_size(ssn)) {
            throw generic_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Subspace size inconsistency detected.");
        }
#endif // LIBTENSOR_DEBUG

        const double *pa = ca.req_const_dataptr(ssn);

        loop_registers<1, 1> r;
        r.m_ptra[0] = pa;
        r.m_ptrb[0] = pb + aoff;
        r.m_ptra_end[0] = pa + dtsa.get_subspace_size(ssn);
        r.m_ptrb_end[0] = pb + dimsb.get_size();

        {
            std::auto_ptr< kernel_base<linalg, 1, 1> >kern(
                kern_dadd1<linalg>::match(1.0, loop_in, loop_out));
            loop_list_runner<linalg, 1, 1>(loop_in).run(0, r, *kern);
        }

        ca.ret_const_dataptr(ssn, pa); pa = 0;
    }

    cb.ret_dataptr(pb); pb = 0;
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONV_DIAG_TENSOR_IMPL_H

