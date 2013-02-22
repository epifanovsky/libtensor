#ifndef LIBTENSOR_TOD_BTCONV_IMPL_H
#define LIBTENSOR_TOD_BTCONV_IMPL_H

#include <list>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/loop_list_runner.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_btconv.h"

namespace libtensor {


template<size_t N>
const char *tod_btconv<N>::k_clazz = "tod_btconv<N>";


template<size_t N>
tod_btconv<N>::tod_btconv(block_tensor_rd_i<N, double> &bt) : m_bt(bt) {

}


template<size_t N>
tod_btconv<N>::~tod_btconv() {

}


template<size_t N>
void tod_btconv<N>::perform(dense_tensor_wr_i<N, double> &t) {
    tod_btconv<N>::start_timer();

    static const char *method = "perform(dense_tensor_wr_i<N, double>&)";

    const block_index_space<N> &bis = m_bt.get_bis();
    dimensions<N> bidims(bis.get_block_index_dims());
    if(!bis.get_dims().equals(t.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    block_tensor_rd_ctrl<N, double> src_ctrl(m_bt);
    dense_tensor_wr_ctrl<N, double> dst_ctrl(t);

    double *dst_ptr = dst_ctrl.req_dataptr();
    size_t sz = t.get_dims().get_size();
    for(register size_t i = 0; i < sz; i++) dst_ptr[i] = 0.0;

    orbit_list<N, double> orblst(src_ctrl.req_const_symmetry());
    typename orbit_list<N, double>::iterator iorbit = orblst.begin();
    for(; iorbit != orblst.end(); iorbit++) {

        index<N> idx;
        orblst.get_index(iorbit, idx);
        orbit<N, double> orb(src_ctrl.req_const_symmetry(), idx);
        abs_index<N> abidx(orb.get_acindex(), bidims);
        if(src_ctrl.req_is_zero_block(abidx.get_index())) continue;

        dense_tensor_rd_i<N, double> &blk =
                src_ctrl.req_const_block(abidx.get_index());
        {
        dense_tensor_rd_ctrl<N, double> blk_ctrl(blk);
        const double *src_ptr = blk_ctrl.req_const_dataptr();

        typename orbit<N, double>::iterator i = orb.begin();
        while(i != orb.end()) {
            abs_index<N> aidx(orb.get_abs_index(i), bidims);
            const tensor_transf<N, double> &tr = orb.get_transf(i);
            index<N> dst_offset = bis.get_block_start(aidx.get_index());
            copy_block(dst_ptr, t.get_dims(), dst_offset,
                src_ptr, blk.get_dims(),
                tr.get_perm(), tr.get_scalar_tr().get_coeff());
            i++;
        }

        blk_ctrl.ret_const_dataptr(src_ptr);
        }
        src_ctrl.ret_const_block(abidx.get_index());

    }

    dst_ctrl.ret_dataptr(dst_ptr);

    tod_btconv<N>::stop_timer();
}


template<size_t N>
void tod_btconv<N>::copy_block(double *optr, const dimensions<N> &odims,
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
        size_t inca = idims.get_increment(i);
        size_t incb = odims.get_increment(ib[i]);
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
        std::auto_ptr< kernel_base<linalg, 1, 1> > kern(
                kern_dcopy<linalg>::match(icoeff, loop_in, loop_out));
        tod_btconv<N>::start_timer(kern->get_name());
        loop_list_runner<linalg, 1, 1>(loop_in).run(0, regs, *kern);
        tod_btconv<N>::stop_timer(kern->get_name());
    }
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_BTCONV_IMPL_H
