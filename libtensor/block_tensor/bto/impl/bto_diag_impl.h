#ifndef LIBTENSOR_BTO_DIAG_IMPL_H
#define LIBTENSOR_BTO_DIAG_IMPL_H

#include <libtensor/core/block_index_subspace_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/symmetry/so_permute.h>

namespace libtensor {


template<size_t N, size_t M, typename Traits>
const char *bto_diag<N, M, Traits>::k_clazz = "bto_diag<N, M, Traits>";


template<size_t N, size_t M, typename Traits>
bto_diag<N, M, Traits>::bto_diag(block_tensora_t &bta, const mask<N> &m,
    const scalar_tr_t &c) :

    m_bta(bta), m_msk(m), m_tr(permutation<k_orderb>(), c),
    m_bis(mk_bis(bta.get_bis(), m_msk)),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

    make_symmetry();
    make_schedule();
}


template<size_t N, size_t M, typename Traits>
bto_diag<N, M, Traits>::bto_diag(block_tensora_t &bta, const mask<N> &m,
    const permutation<N - M + 1> &p, const scalar_tr_t &c) :

    m_bta(bta), m_msk(m), m_tr(p, c),
    m_bis(mk_bis(bta.get_bis(), m_msk).permute(p)),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims())  {

    make_symmetry();
    make_schedule();
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::sync_on() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    ctrla.req_sync_on();
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::sync_off() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    ctrla.req_sync_off();
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::compute_block(bool zero, blockb_t &blk,
        const index<k_orderb> &ib, const tensorb_tr_t &trb,
        const scalar_tr_t &c, cpu_pool &cpus) {

    compute_block(blk, ib, trb, zero, c, cpus);
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::compute_block(blockb_t &blk,
    const index<k_orderb> &ib, const tensorb_tr_t &trb,
    bool zero, const scalar_tr_t &c, cpu_pool &cpus) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;
    typedef typename Traits::template to_diag_type<N, M>::type to_diag_t;

    bto_diag<N, M, Traits>::start_timer();

    try {

        block_tensor_ctrl_t ctrla(m_bta);
        dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

        //  Build ia from ib
        //
        sequence<k_ordera, size_t> map(0);
        size_t j = 0, jd; // Current index, index on diagonal
        bool b = false;
        for(size_t i = 0; i < k_ordera; i++) {
            if(m_msk[i]) {
                if(!b) { map[i] = jd = j++; b = true; }
                else { map[i] = jd; }
            } else {
                map[i] = j++;
            }
        }
        index<k_ordera> ia;
        index<k_orderb> ib2(ib);
        permutation<k_orderb> pinvb(m_tr.get_perm(), true);
        ib2.permute(pinvb);
        for(size_t i = 0; i < k_ordera; i++) ia[i] = ib2[map[i]];

        //  Find canonical index cia, transformation cia->ia
        //
        orbit<k_ordera, element_t> oa(ctrla.req_const_symmetry(), ia);
        abs_index<k_ordera> acia(oa.get_abs_canonical_index(), bidimsa);
        const tensora_tr_t &tra = oa.get_transf(ia);

        //  Build new diagonal mask and permutation in b
        //
        mask<k_ordera> m1(m_msk), m2(m_msk);
        sequence<k_ordera, size_t> map1(map), map2(map);
        m2.permute(tra.get_perm());
        tra.get_perm().apply(map2);

        sequence<N - M, size_t> seq1(0), seq2(0);
        sequence<k_orderb, size_t> seqb1(0), seqb2(0);
        for(register size_t i = 0, j1 = 0, j2 = 0; i < k_ordera; i++) {
            if(!m1[i]) seq1[j1++] = map1[i];
            if(!m2[i]) seq2[j2++] = map2[i];
        }
        bool b1 = false, b2 = false;
        for(register size_t i = 0, j1 = 0, j2 = 0; i < k_orderb; i++) {
            if(m1[i] && !b1) { seqb1[i] = k_orderb; b1 = true; }
            else { seqb1[i] = seq1[j1++]; }
            if(m2[i] && !b2) { seqb2[i] = k_orderb; b2 = true; }
            else { seqb2[i] = seq2[j2++]; }
        }

        permutation_builder<k_orderb> pb(seqb2, seqb1);
        permutation<k_orderb> permb(pb.get_perm());
        permb.permute(m_tr.get_perm());
        permb.permute(permutation<k_orderb>(trb.get_perm(), true));

        //  Invoke the tensor operation
        //
        blocka_t &blka = ctrla.req_block(acia.get_index());

        scalar_tr_t sa(tra.get_scalar_tr());
        sa.invert().transform(m_tr.get_scalar_tr());
        sa.transform(trb.get_scalar_tr());

        if(zero) {
            sa.transform(c);
            to_diag_t(blka, m2, permb, sa.get_coeff()).perform(blk);
        }
        else {
            to_diag_t(blka, m2, permb,
                    sa.get_coeff()).perform(blk, c.get_coeff());
        }
        ctrla.ret_block(acia.get_index());

    }
    catch (...) {
        bto_diag<N, M, Traits>::stop_timer();
        throw;
    }

    bto_diag<N, M, Traits>::stop_timer();

}


template<size_t N, size_t M, typename Traits>
block_index_space<N - M + 1> bto_diag<N, M, Traits>::mk_bis(
    const block_index_space<N> &bis, const mask<N> &msk) {

    static const char *method =
        "mk_bis(const block_index_space<N>&, const mask<N>&)";

    //  Create the mask for the subspace builder
    //
    mask<N> m;
    bool b = false;
    for(size_t i = 0; i < N; i++) {
        if(msk[i]) {
            if(!b) { m[i] = true; b = true; }
        } else {
            m[i] = true;
        }
    }

    //  Build the output block index space
    //
    block_index_subspace_builder<N - M + 1, M - 1> bb(bis, m);
    block_index_space<k_orderb> obis(bb.get_bis());
    obis.match_splits();

    return obis;
}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::make_symmetry() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ca(m_bta);

    block_index_space<k_orderb> bis(m_bis);
    permutation<k_orderb> pinv(m_tr.get_perm(), true);
    bis.permute(pinv);
    symmetry<k_orderb, element_t> symx(bis);
    so_merge<N, M - 1, element_t>(ca.req_const_symmetry(),
            m_msk, sequence<N, size_t>()).perform(symx);
    so_permute<k_orderb, element_t>(symx, m_tr.get_perm()).perform(m_sym);

}


template<size_t N, size_t M, typename Traits>
void bto_diag<N, M, Traits>::make_schedule() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<k_orderb> pinv(m_tr.get_perm(), true);
    size_t map[k_ordera];
    size_t j = 0, jd;
    bool b = false;
    for(size_t i = 0; i < k_ordera; i++) {
        if(m_msk[i]) {
            if(b) map[i] = jd;
            else { map[i] = jd = j++; b = true; }
        } else {
            map[i] = j++;
        }
    }

    orbit_list<k_ordera, element_t> ola(ctrla.req_const_symmetry());
    orbit_list<k_orderb, element_t> olb(m_sym);
    for (typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
            iob != olb.end(); iob++) {

        index<k_ordera> idxa;
        index<k_orderb> idxb(olb.get_index(iob));
        idxb.permute(pinv);

        for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];

        orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);
        if(! ola.contains(oa.get_abs_canonical_index())) continue;

        abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(), bidimsa);

        if(ctrla.req_is_zero_block(cidxa.get_index())) continue;

        m_sch.insert(olb.get_abs_index(iob));
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_DIAG_IMPL_H
