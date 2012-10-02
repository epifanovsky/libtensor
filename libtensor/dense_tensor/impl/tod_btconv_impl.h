#ifndef LIBTENSOR_TOD_BTCONV_IMPL_H
#define LIBTENSOR_TOD_BTCONV_IMPL_H

#include <libtensor/linalg/linalg.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
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

        orbit<N, double> orb(src_ctrl.req_const_symmetry(),
            orblst.get_index(iorbit));
        abs_index<N> abidx(orb.get_abs_canonical_index(), bidims);
        if(src_ctrl.req_is_zero_block(abidx.get_index())) continue;

        dense_tensor_i<N, double> &blk =
                src_ctrl.req_const_block(abidx.get_index());
        {
        dense_tensor_rd_ctrl<N, double> blk_ctrl(blk);
        const double *src_ptr = blk_ctrl.req_const_dataptr();

        typename orbit<N, double>::iterator i = orb.begin();
        while(i != orb.end()) {
            abs_index<N> aidx(i->first, bidims);
            const tensor_transf<N, double> &tr = i->second;
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

    loop_list_t lst;
    for(size_t i = 0; i < N; i++) {
        size_t inca = idims.get_increment(i);
        size_t incb = odims.get_increment(ib[i]);
        loop_list_node node(idims[i], inca, incb);
        if(i == N - 1) {
            node.m_op = new op_dcopy(idims[i], inca, incb, icoeff);
        } else {
            node.m_op = new op_loop(idims[i], inca, incb);
        }
        lst.push_back(node);
    }

    registers regs;
    regs.m_ptra = iptr;
    regs.m_ptrb = optr + abs_index<N>::get_abs_index(ooffs, odims);
    processor_t proc(lst, regs);
    proc.process_next();

    for(typename loop_list_t::iterator i = lst.begin();
        i != lst.end(); i++) {

        delete i->m_op;
        i->m_op = NULL;
    }
}


template<size_t N>
void tod_btconv<N>::op_loop::exec(processor_t &proc, registers &regs)
    throw(exception) {

    const double *ptra = regs.m_ptra;
    double *ptrb = regs.m_ptrb;

    for(size_t i=0; i<m_len; i++) {
        regs.m_ptra = ptra;
        regs.m_ptrb = ptrb;
        proc.process_next();
        ptra += m_inca;
        ptrb += m_incb;
    }
}


template<size_t N>
void tod_btconv<N>::op_dcopy::exec(processor_t &proc, registers &regs)
    throw(exception) {
    linalg::copy_i_i(0, m_len, regs.m_ptra, m_inca, regs.m_ptrb, m_incb);
    if(m_c != 1.0) linalg::mul1_i_x(0, m_len, m_c, regs.m_ptrb, m_incb);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_BTCONV_IMPL_H
