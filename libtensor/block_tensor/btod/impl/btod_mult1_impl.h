#ifndef LIBTENSOR_BTOD_MULT1_IMPL_H
#define LIBTENSOR_BTOD_MULT1_IMPL_H

#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_mult1.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/btod/bad_block_index_space.h>
#include "../btod_mult1.h"

namespace libtensor {


template<size_t N>
const char *btod_mult1<N>::k_clazz = "btod_mult1<N>";


template<size_t N>
btod_mult1<N>::btod_mult1(
        block_tensor_i<N, double> &btb, bool recip, double c) :

    m_btb(btb), m_recip(recip), m_c(c) {
}

template<size_t N>
btod_mult1<N>::btod_mult1(
        block_tensor_i<N, double> &btb, const permutation<N> &pb,
        bool recip, double c) :

    m_btb(btb), m_pb(pb), m_recip(recip), m_c(c) {
}

template<size_t N>
void btod_mult1<N>::perform(block_tensor_i<N, double> &bta) {

    static const char *method = "perform(block_tensor_i<N, double>&)";

    if(!bta.get_bis().equals(m_btb.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "bta");
    }

    do_perform(bta, true, 1.0);
}


template<size_t N>
void btod_mult1<N>::perform(block_tensor_i<N, double> &bta, double c) {

    static const char *method =
        "perform(block_tensor_i<N, double>&, double)";

    if(!bta.get_bis().equals(m_btb.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "bta");
    }

    do_perform(bta, false, c);
}


template<size_t N>
void btod_mult1<N>::do_perform(
    block_tensor_i<N, double> &bta, bool zero, double c) {

    static const char *method =
        "do_perform(block_tensor_i<N, double>&, bool, double)";

    btod_mult1::start_timer();

    block_tensor_ctrl<N, double> ctrla(bta), ctrlb(m_btb);

    // Copy sym(A) and permuted sym(B) and install \sym(A) \cap \sym(B) in A

    symmetry<N, double> syma(bta.get_bis());
    so_copy<N, double>(ctrla.req_const_symmetry()).perform(syma);

    sequence<N + N, size_t> seq1b, seq2b;
    for (size_t i = 0; i < N; i++) {
        seq1b[i] = seq2b[i] = i;
    }
    for (size_t i = N, j = 0; i < N + N; i++, j++) {
        seq1b[i] = i; seq2b[i] = m_pb[j] + N;
    }
    permutation_builder<N + N> pbb(seq2b, seq1b);

    block_index_space_product_builder<N, N> bbx(bta.get_bis(), bta.get_bis(),
            permutation<N + N>());

    symmetry<N + N, double> symx(bbx.get_bis());
    so_dirprod<N, N, double>(syma,
            ctrlb.req_const_symmetry(), pbb.get_perm()).perform(symx);
    mask<N + N> msk;
    sequence<N + N, size_t> seq;
    for (register size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, double>(symx, msk, seq).perform(ctrla.req_symmetry());

    // First loop over all orbits in sym(A) \cap sym(B) and copy blocks which
    // were not canonical in sym(A)

    orbit_list<N, double> ol(ctrla.req_symmetry());

    for(typename orbit_list<N, double>::iterator io = ol.begin();
        io != ol.end(); io++) {

        index<N> idx(ol.get_index(io));

        orbit<N, double> oa(syma, idx);
        abs_index<N> cidxa(oa.get_abs_canonical_index(),
                bta.get_bis().get_block_index_dims());

        if (idx.equals(cidxa.get_index()))
            continue;

        if (ctrla.req_is_zero_block(cidxa.get_index()))
            continue;

        dense_tensor_i<N, double> &blk = ctrla.req_block(idx);
        dense_tensor_i<N, double> &blka = ctrla.req_block(cidxa.get_index());

        const tensor_transf<N, double> &tra = oa.get_transf(idx);

        tod_copy<N>(blka, tra.get_perm(), tra.get_scalar_tr().get_coeff()).
            perform(true, 1.0, blk);

        ctrla.ret_block(cidxa.get_index());
        ctrla.ret_block(idx);
    }

    // Second loop over all orbits in sym(A) \cap sym(B) and do the operation

    permutation<N> pinvb(m_pb, true);

    for(typename orbit_list<N, double>::iterator ioa = ol.begin();
        ioa != ol.end(); ioa++) {

        index<N> idxa(ol.get_index(ioa)), idxb(idxa);

        idxb.permute(pinvb);
        orbit<N, double> ob(ctrlb.req_const_symmetry(), idxb);
        abs_index<N> cidxb(ob.get_abs_canonical_index(),
                m_btb.get_bis().get_block_index_dims());

        bool zeroa = ctrla.req_is_zero_block(idxa);
        bool zerob = ctrlb.req_is_zero_block(cidxb.get_index());

        if(m_recip && zerob) {
            throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "zero in btb");
        }

        if(zero && (zeroa || zerob)) {
            ctrla.req_zero_block(idxa);
            continue;
        }

        if (zeroa || zerob)
            continue;

        dense_tensor_i<N, double> &blka = ctrla.req_block(idxa);
        dense_tensor_i<N, double> &blkb = ctrlb.req_block(cidxb.get_index());

        const tensor_transf<N, double> &trb = ob.get_transf(idxb);
        double k = m_c;
        if (m_recip) k /= trb.get_scalar_tr().get_coeff();
        else k *= trb.get_scalar_tr().get_coeff();

        permutation<N> pb(trb.get_perm());
        pb.permute(m_pb);

        if(zero) {
            tod_mult1<N>(blkb, pb, m_recip, k * c).perform(blka);
        } else {
            tod_mult1<N>(blkb, pb, m_recip, k).perform(blka, c);
        }

        ctrla.ret_block(idxa);
        ctrlb.ret_block(cidxb.get_index());
    }

    btod_mult1::stop_timer();

}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT1_IMPL_H
