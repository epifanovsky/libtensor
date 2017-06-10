#ifndef LIBTENSOR_BTOD_EXPORT_IMPL_H
#define LIBTENSOR_BTOD_EXPORT_IMPL_H

#include <list>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/core/orbit.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "../btod_export.h"

namespace libtensor {


template<size_t N>
const char btod_export<N>::k_clazz[] = "btod_export<N>";


template<size_t N>
btod_export<N>::btod_export(gen_block_tensor_rd_i<N, bti_traits> &bt) :
    m_bt(bt) {

}


template<size_t N>
btod_export<N>::~btod_export() {

}


template<size_t N>
void btod_export<N>::perform(double *ptr) {

    static const char method[] = "perform(dense_tensor_wr_i<N, double>&)";

    const block_index_space<N> &bis = m_bt.get_bis();
    dimensions<N> bidims(bis.get_block_index_dims());
    const dimensions<N> &dims = bis.get_dims();

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(m_bt);

    size_t sz = dims.get_size();
    for(size_t i = 0; i < sz; i++) ptr[i] = 0.0;

    std::vector<size_t> nzblk;
    ctrl.req_nonzero_blocks(nzblk);

    for(size_t i = 0; i < nzblk.size(); i++) {

        index<N> idx;
        abs_index<N>::get_index(nzblk[i], bidims, idx);
        orbit<N, double> o(ctrl.req_const_symmetry(), idx);

        dense_tensor_rd_i<N, double> &blk = ctrl.req_const_block(idx);
        {
            dense_tensor_rd_ctrl<N, double> tctrl(blk);
            const double *sptr = tctrl.req_const_dataptr();

            for(typename orbit<N, double>::iterator j = o.begin();
                    j != o.end(); ++j) {

                abs_index<N> aidx(o.get_abs_index(j), bidims);
                const tensor_transf<N, double> &tr = o.get_transf(j);
                index<N> offset = bis.get_block_start(aidx.get_index());
                copy_block(ptr, dims, offset, sptr, blk.get_dims(),
                    tr.get_perm(), tr.get_scalar_tr().get_coeff());
            }

            tctrl.ret_const_dataptr(sptr);
        }
        ctrl.ret_const_block(idx);
    }
}


template<size_t N>
void btod_export<N>::copy_block(double *optr, const dimensions<N> &odims,
    const index<N> &ooffs, const double *iptr, const dimensions<N> &idims,
    const permutation<N> &iperm, double icoeff) {

    permutation<N> inv_perm(iperm);
    inv_perm.invert();
    sequence<N, size_t> ib(0);
    for(size_t i = 0; i < N; i++) ib[i] = i;
    inv_perm.apply(ib);

    std::list< loop_list_node<1, 1> > loop_in, loop_out;
    typename std::list< loop_list_node<1, 1> >::iterator inode = loop_in.end();

    for(size_t i = 0; i < N; i++) {
        inode = loop_in.insert(loop_in.end(), loop_list_node<1, 1>(idims[i]));
        inode->stepa(0) = idims.get_increment(i);
        inode->stepb(0) = odims.get_increment(ib[i]);
    }

    double *pb = optr + abs_index<N>::get_abs_index(ooffs, odims);

    loop_registers<1, 1> regs;
    regs.m_ptra[0] = iptr;
    regs.m_ptrb[0] = pb;
    regs.m_ptra_end[0] = iptr + idims.get_size();
    regs.m_ptrb_end[0] = pb + odims.get_size();

    {
        std::auto_ptr< kernel_base<linalg, 1, 1, double> > kern(
                kern_dcopy<linalg>::match(icoeff, loop_in, loop_out));
        loop_list_runner<linalg, 1, 1>(loop_in).run(0, regs, *kern);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXPORT_IMPL_H
