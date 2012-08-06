#ifndef LIBTENSOR_BTOD_EXTRACT_IMPL_H
#define LIBTENSOR_BTOD_EXTRACT_IMPL_H

#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/dense_tensor/tod_extract.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/symmetry/so_reduce.h>
#include <libtensor/symmetry/so_permute.h>
#include <libtensor/btod/bad_block_index_space.h>
#include <libtensor/not_implemented.h>
#include "../btod_extract.h"

namespace libtensor {


template<size_t N, size_t M>
const char *btod_extract<N, M>::k_clazz = "btod_extract<N, M>";


template<size_t N, size_t M>
btod_extract<N, M>::btod_extract(block_tensor_i<N, double> &bta,
    const mask<N> &m, const index<N> &idxbl, const index<N> &idxibl,
    double c) :

    m_bta(bta), m_msk(m), m_idxbl(idxbl), m_idxibl(idxibl), m_c(c),
    m_bis(mk_bis(bta.get_bis(), m_msk, permutation<N - M>())), m_sym(m_bis),
    m_sch(m_bis.get_block_index_dims()) {

    block_tensor_ctrl<N, double> ctrla(bta);

    sequence<N, size_t> seq(0);
    mask<N> invmsk;
    for (register size_t i = 0, j = 0; i < N; i++) {
        invmsk[i] = !m_msk[i];
        if (invmsk[i]) seq[i] = j++;
    }

    so_reduce<N, M, double>(ctrla.req_const_symmetry(),
            invmsk, seq, index_range<N>(idxbl, idxbl),
            index_range<N>(idxibl, idxibl)).perform(m_sym);

    make_schedule();
}


template<size_t N, size_t M>
btod_extract<N, M>::btod_extract(block_tensor_i<N, double> &bta,
    const mask<N> &m, const permutation<N - M> &perm,
    const index<N> &idxbl, const index<N> &idxibl, double c) :

    m_bta(bta), m_msk(m), m_perm(perm), m_idxbl(idxbl), m_idxibl(idxibl),
    m_c(c), m_bis(mk_bis(bta.get_bis(), m_msk, perm)), m_sym(m_bis),
    m_sch(m_bis.get_block_index_dims()) {

    permutation<N - M> pinv(perm, true);
    block_index_space<N - M> bisinv(m_bis);
    bisinv.permute(pinv);

    block_tensor_ctrl<N, double> ctrla(bta);
    symmetry<k_orderb, double> sym(bisinv);

    sequence<N, size_t> seq(0);
    mask<N> invmsk;
    for (register size_t i = 0, j = 0; i < N; i++) {
        invmsk[i] = !m_msk[i];
        if (invmsk[i]) seq[i] = j++;
    }

    so_reduce<N, M, double>(ctrla.req_const_symmetry(),
            invmsk, seq, index_range<N>(idxbl, idxbl),
            index_range<N>(idxibl, idxibl)).perform(sym);
    so_permute<k_orderb, double>(sym, perm).perform(m_sym);

    make_schedule();
}


template<size_t N, size_t M>
void btod_extract<N, M>::sync_on() {

    block_tensor_ctrl<N, double> ctrla(m_bta);
    ctrla.req_sync_on();
}


template<size_t N, size_t M>
void btod_extract<N, M>::sync_off() {

    block_tensor_ctrl<N, double> ctrla(m_bta);
    ctrla.req_sync_off();
}


template<size_t N, size_t M>
void btod_extract<N, M>::perform(bto_stream_i<N - M, btod_traits> &out) {

    throw not_implemented(g_ns, k_clazz, "perform(bto_stream_i&)",
        __FILE__, __LINE__);
}


template<size_t N, size_t M>
void btod_extract<N, M>::compute_block(bool zero,
    dense_tensor_i<k_orderb, double> &blk, const index<k_orderb> &idx,
    const tensor_transf<k_orderb, double> &tr, const double &c) {

    do_compute_block(blk, idx, tr, c, zero);
}


template<size_t N, size_t M>
void btod_extract<N, M>::do_compute_block(dense_tensor_i<k_orderb, double> &blk,
    const index<k_orderb> &idx, const tensor_transf<k_orderb, double> &tr,
    double c, bool zero) {

    btod_extract<N, M>::start_timer();

    block_tensor_ctrl<k_ordera, double> ctrla(m_bta);

    permutation<k_orderb> pinv(m_perm, true);

    index<k_ordera> idxa;
    index<k_orderb> idxb(idx);

    idxb.permute(pinv);

    for(size_t i = 0, j = 0; i < k_ordera; i++) {
        if(m_msk[i]) {
            idxa[i] = idxb[j++];
        } else {
            idxa[i] = m_idxbl[i];
        }
    }

    orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);

    abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(),
            m_bta.get_bis().get_block_index_dims());
    tensor_transf<k_ordera, double> tra(oa.get_transf(idxa)); tra.invert();

    mask<k_ordera> msk1(m_msk), msk2(m_msk);
    msk2.permute(tra.get_perm());

    sequence<k_ordera, size_t> seqa1(0), seqa2(0);
    sequence<k_orderb, size_t> seqb1(0), seqb2(0);
    for(register size_t i = 0; i < k_ordera; i++) seqa2[i] = seqa1[i] = i;
    tra.get_perm().apply(seqa2);
    for(register size_t i = 0, j1 = 0, j2 = 0; i < k_ordera; i++) {
        if(msk1[i]) seqb1[j1++] = seqa1[i];
        if(msk2[i]) seqb2[j2++] = seqa2[i];
    }

    permutation_builder<k_orderb> pb(seqb2, seqb1);
    permutation<k_orderb> permb(pb.get_perm());
    permb.permute(m_perm);
    permb.permute(tr.get_perm());

    index<k_ordera> idxibl2(m_idxibl);
    idxibl2.permute(tra.get_perm());

    bool zeroa = !oa.is_allowed();
    if(!zeroa) zeroa = ctrla.req_is_zero_block(cidxa.get_index());

    if(!zeroa) {

        dense_tensor_i<k_ordera, double> &blka = ctrla.req_block(
            cidxa.get_index());
        if(zero) {
            tod_extract<N, M>(blka, msk2, permb, idxibl2,
                tra.get_scalar_tr().get_coeff() * m_c * c).perform(blk);
        } else {
            tod_extract<N, M>(blka, msk2, permb, idxibl2,
                tra.get_scalar_tr().get_coeff() * m_c).perform(blk, c);
        }
        ctrla.ret_block(cidxa.get_index());
    } else {

        if(zero) tod_set<N - M>().perform(blk);
    }

    btod_extract<N, M>::stop_timer();
}


template<size_t N, size_t M>
block_index_space<N - M> btod_extract<N, M>::mk_bis(
    const block_index_space<N> &bis, const mask<N> &msk,
    const permutation<N - M> &perm) {

    static const char *method = "mk_bis(const block_index_space<N>&, "
        "const mask<N>&, const permutation<N - M>&)";

    dimensions<N> idims(bis.get_dims());

    //  Compute output dimensions
    //

    index<k_orderb> i1, i2;

    size_t m = 0, j = 0;
    size_t map[k_orderb];//map between B and A

    for(size_t i = 0; i < N; i++) {
        if(msk[i]) {
            i2[j] = idims[i] - 1;
            map[j] = i;
            j++;
        } else {
            m++;
        }
    }


    if(m != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "m");
    }

    block_index_space<k_orderb> obis(dimensions<k_orderb>(
        index_range<k_orderb>(i1, i2)));

    mask<k_orderb> msk_done;
    bool done = false;
    while(!done) {

        size_t i = 0;
        while(i < k_orderb && msk_done[i]) i++;
        if(i == k_orderb) {
            done = true;
            continue;
        }
        size_t typ = bis.get_type(map[i]);
        const split_points &splits = bis.get_splits(typ);
        mask<k_orderb> msk_typ;
        for(size_t k = 0; k < k_orderb; k++) {
            if(bis.get_type(map[k]) == typ) msk_typ[k] = true;
        }
        size_t npts = splits.get_num_points();
        for(register size_t k = 0; k < npts; k++) {
            obis.split(msk_typ, splits[k]);
        }
        msk_done |= msk_typ;
    }

    obis.permute(perm);
    return obis;
}


template<size_t N, size_t M>
void btod_extract<N, M>::make_schedule() {

    btod_extract<N, M>::start_timer("make_schedule");

    block_tensor_ctrl<N, double> ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<k_orderb> pinv(m_perm, true);

    orbit_list<k_orderb, double> olb(m_sym);
    for (typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
            iob != olb.end(); iob++) {

        index<k_ordera> idxa;
        index<k_orderb> idxb(olb.get_index(iob));

        idxb.permute(pinv);

        for(size_t i = 0, j = 0; i < k_ordera; i++) {
            if(m_msk[i]) idxa[i] = idxb[j++];
            else idxa[i] = m_idxbl[i];
        }

        orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);

        abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(),
                m_bta.get_bis().get_block_index_dims());

        if(!oa.is_allowed()) continue;
        if(ctrla.req_is_zero_block(cidxa.get_index())) continue;

        m_sch.insert(olb.get_abs_index(iob));
    }

    btod_extract<N, M>::stop_timer("make_schedule");

}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXTRACT_IMPL_H
