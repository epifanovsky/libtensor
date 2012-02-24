#ifndef LIBTENSOR_TOD_CONV_DIAG_TENSOR_IMPL_H
#define LIBTENSOR_TOD_CONV_DIAG_TENSOR_IMPL_H

#include <list>
#include <vector>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/tod/kernels/loop_list_runner.h>
#include <libtensor/tod/kernels/kern_add_generic.h>
#include "../diag_tensor_ctrl.h"
#include "../tod_conv_diag_tensor.h"

namespace libtensor {


template<size_t N>
void tod_conv_diag_tensor<N>::perform(dense_tensor_wr_i<N, double> &tb) {

    const dimensions<N> &dims = m_ta.get_dims();

    if(!dims.equals(tb.get_dims())) {
        throw 0;
    }

    const diag_tensor_space<N> &dtsa = m_ta.get_space();
    diag_tensor_rd_ctrl<N, double> ca(m_ta);
    dense_tensor_wr_ctrl<N, double> cb(tb);

    double *pb = cb.req_dataptr();

    {
        //  Zero output array
        size_t sz = dims.get_size();
        for(size_t i = 0; i < sz; i++) pb[i] = 0.0;
    }

    double zero = 0.0;

    std::vector<size_t> ssl; // List of subspaces
    dtsa.get_all_subspaces(ssl);
    for(size_t ssi = 0; ssi < ssl.size(); ssi++) {

        size_t ssn = ssl[ssi];

        const diag_tensor_subspace<N> &ss = dtsa.get_subspace(ssn);

        std::list< loop_list_node<2, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<2, 1> >::iterator inode =
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
                    if(m[j]) stepb += dims.get_increment(j);
                }
                mdone |= m;
            } else {
                stepb = dims.get_increment(i);
                mdone[i] = true;
            }
            size_t w = dims[i];
            for(typename std::list< loop_list_node<2, 1> >::iterator jnode =
                loop_in.begin(); jnode != loop_in.end(); ++jnode) {
                jnode->stepa(0) *= w;
            }
            inode = loop_in.insert(loop_in.end(), loop_list_node<2, 1>(w));
            inode->stepa(0) = 1;
            inode->stepa(1) = 0;
            inode->stepb(0) = stepb;
        }
#ifdef LIBTENSOR_DEBUG
        if(loop_in.begin()->stepa(0) * loop_in.begin()->weight() !=
            dtsa.get_subspace_size(ssn)) {
            throw 0;
        }
#endif // LIBTENSOR_DEBUG

        const double *pa = ca.req_const_dataptr(ssn);

        loop_registers<2, 1> r;
        r.m_ptra[0] = pa;
        r.m_ptra[1] = &zero;
        r.m_ptrb[0] = pb;
        r.m_ptra_end[0] = pa + dtsa.get_subspace_size(ssn);
        r.m_ptra_end[1] = &zero + 1;
        r.m_ptrb_end[0] = pb + dims.get_size();

        {
            std::auto_ptr< kernel_base<2, 1> >kern(
                kern_add_generic::match(1.0, 1.0, 1.0, loop_in, loop_out));
            loop_list_runner<2, 1>(loop_in).run(r, *kern);
        }

        ca.ret_const_dataptr(ssn, pa); pa = 0;
    }

    cb.ret_dataptr(pb); pb = 0;
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONV_DIAG_TENSOR_IMPL_H

