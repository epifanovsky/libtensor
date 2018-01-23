#ifndef LIBTENSOR_GEN_BTO_DIAG_IMPL_H
#define LIBTENSOR_GEN_BTO_DIAG_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/block_index_subspace_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/symmetry/so_permute.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_diag.h"

namespace libtensor {


template<size_t N, size_t M, typename Traits, typename Timed>
const char *gen_bto_diag<N, M, Traits, Timed>::k_clazz =
        "gen_bto_diag<N, M, Traits, Timed>";


template<size_t N, size_t M, typename Traits, typename Timed>
class gen_bto_diag_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<M>::type
        temp_block_tensor_type;

private:
    gen_bto_diag<N, M, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btb;
    index<M> m_idx;
    gen_block_stream_i<M, bti_traits> &m_out;

public:
    gen_bto_diag_task(
        gen_bto_diag<N, M, Traits, Timed> &bto,
        temp_block_tensor_type &btb,
        const index<M> &idx,
        gen_block_stream_i<M, bti_traits> &out);

    virtual ~gen_bto_diag_task() { }
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

};


template<size_t N, size_t M, typename Traits, typename Timed>
class gen_bto_diag_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<M>::type
        temp_block_tensor_type;

private:
    gen_bto_diag<N, M, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btb;
    gen_block_stream_i<M, bti_traits> &m_out;
    const assignment_schedule<M, element_type> &m_sch;
    typename assignment_schedule<M, element_type>::iterator m_i;

public:
    gen_bto_diag_task_iterator(
        gen_bto_diag<N, M, Traits, Timed> &bto,
        temp_block_tensor_type &btb,
        gen_block_stream_i<M, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, typename Traits>
class gen_bto_diag_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, typename Traits, typename Timed>
gen_bto_diag<N, M, Traits, Timed>::gen_bto_diag(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const sequence<N, size_t> &m,
        const tensor_transf_type &tr) :

    m_bta(bta), m_msk(m), m_tr(tr),
    m_bis(mk_bis(bta.get_bis(), m_msk).permute(tr.get_perm())),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

    make_symmetry();
    make_schedule();
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_diag<N, M, Traits, Timed>::perform(
        gen_block_stream_i<NB, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<M>::type
        temp_block_tensor_type;

    gen_bto_diag::start_timer();

    try {

        temp_block_tensor_type btb(m_bis);

        gen_bto_diag_task_iterator<N, M, Traits, Timed> ti(*this, btb, out);
        gen_bto_diag_task_observer<N, M, Traits> to;
        libutil::thread_pool::submit(ti, to);

    } catch(...) {
        gen_bto_diag::stop_timer();
        throw;
    }

    gen_bto_diag::stop_timer();
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_diag<N, M, Traits, Timed>::compute_block(
        bool zero,
        const index<NB> &ib,
        const tensor_transf_type &trb,
        wr_block_type &blkb) {

    gen_bto_diag::start_timer("compute_block");

    try {

        compute_block_untimed(zero, ib, trb, blkb);

    } catch (...) {
        gen_bto_diag::stop_timer("compute_block");
        throw;
    }

    gen_bto_diag::stop_timer("compute_block");
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_diag<N, M, Traits, Timed>::compute_block_untimed(
        bool zero,
        const index<NB> &ib,
        const tensor_transf_type &trb,
        wr_block_type &blkb) {

    typedef typename Traits::template to_diag_type<N, M>::type to_diag;

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    //  Build ia from ib
    //
    sequence<NA, size_t> map(0);
    sequence<NB, size_t> k(NB);
    size_t j = 0, jd = 0; // Current index and j-th diagonal
    for(size_t i = 0; i < NA; i++) {
        if(m_msk[i] != 0) {
            jd = m_msk[i] - 1;
            if(k[jd] == NB) { map[i] = k[jd] = j++; }
            else { map[i] = k[jd]; }
        }
        else {
            map[i] = j++;
        }
    }

    index<NA> ia;
    index<NB> ib2(ib);
    permutation<NB> pinvb(m_tr.get_perm(), true);
    ib2.permute(pinvb);
    for(size_t i = 0; i < NA; i++) ia[i] = ib2[map[i]];

    //  Find canonical index cia, transformation cia->ia
    //
    orbit<NA, element_type> oa(ctrla.req_const_symmetry(), ia);
    abs_index<NA> acia(oa.get_acindex(), bidimsa);
    const tensor_transf<NA, element_type> &tra = oa.get_transf(ia);

    //  Build new diagonal mask and permutation in b
    //
    sequence<NA, size_t> m1(m_msk), m2(m_msk);
    sequence<NA, size_t> map1(map), map2(map);
    tra.get_perm().apply(m2);
    tra.get_perm().apply(map2);

    sequence<NB, size_t> seq1(0), seq2(0);
    sequence<NB, size_t> seqb1(0), seqb2(0);
    for(register size_t i = 0, j1 = 0, j2 = 0; i < NA; i++) {
        if(m1[i] == 0) seq1[j1++] = map1[i];
        if(m2[i] == 0) seq2[j2++] = map2[i];
    }
    mask<NB + 1> b1, b2;
    for(register size_t i = 0, j1 = 0, j2 = 0; i < NB; i++) {
        if (m1[i] != 0 && !b1[m1[i]]) {
            seqb1[i] = NB + m1[i]; b1[m1[i]] = true;
        }
        else { seqb1[i] = seq1[j1++]; }
        if(m2[i] != 0 && !b2[m2[i]]) {
            seqb2[i] = NB + m2[i]; b2[m2[i]] = true;
        }
        else { seqb2[i] = seq2[j2++]; }
    }

    permutation_builder<NB> pb(seqb1, seqb2);
    permutation<NB> permb(pb.get_perm());
    permb.permute(m_tr.get_perm());
    permb.permute(permutation<NB>(trb.get_perm(), true));

    scalar_transf<element_type> sa(tra.get_scalar_tr());
    sa.invert().transform(m_tr.get_scalar_tr());
    sa.transform(trb.get_scalar_tr());

    tensor_transf<NB, element_type> tr(permb, sa);

    //  Invoke the tensor operation
    //
    rd_block_type &blka = ctrla.req_const_block(acia.get_index());
    to_diag(blka, m2, tr).perform(zero, blkb);
    ctrla.ret_const_block(acia.get_index());
}


template<size_t N, size_t M, typename Traits, typename Timed>
block_index_space<M> gen_bto_diag<N, M, Traits, Timed>::mk_bis(
    const block_index_space<NA> &bis, const sequence<NA, size_t> &msk) {

    static const char method[] = "mk_bis(const block_index_space<NA> &, "
            "const sequence<NA, size_t> &)";

    //  Create the mask for the subspace builder
    //
    mask<NA> m;
    mask<NB> b;
    bool bad_msk = false;
    for(size_t i = 0, id = 0; i < N; i++) {
        if (msk[i] != 0) {
            id = msk[i] - 1;
            if (id >= NB) { bad_msk = true; break; }
            if (! b[id]) { m[i] = true; b[id] = true; }
        }
        else {
            m[i] = true;
        }
    }
    if (bad_msk) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "msk");
    }

    //  Build the output block index space
    //
    block_index_subspace_builder<NB, NA - NB> bb(bis, m);
    block_index_space<NB> obis(bb.get_bis());
    obis.match_splits();

    return obis;
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_diag<N, M, Traits, Timed>::make_symmetry() {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);

    block_index_space<NB> bis(m_bis);
    permutation<NB> pinv(m_tr.get_perm(), true);
    bis.permute(pinv);

    mask<NA> m;
    sequence<NA, size_t> s(0);
    for (size_t i = 0; i < NA; i++) {
        if (m_msk[i] != 0) { m[i] = true; s[i] = m_msk[i] - 1; }
    }

    symmetry<M, element_type> symx(bis);
    so_merge<N, N - M, element_type>(
            ca.req_const_symmetry(), m, s).perform(symx);
    so_permute<M, element_type>(symx, m_tr.get_perm()).perform(m_sym);

}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_diag<N, M, Traits, Timed>::make_schedule() {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ctrla(m_bta);
    dimensions<NA> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<NB> pinv(m_tr.get_perm(), true);
    size_t map[NA], id = 0;
    sequence<NB, size_t> jb(NB);
    for(size_t i = 0, j = 0; i < NA; i++) {
        if (m_msk[i] != 0) {
            id = m_msk[i] - 1;
            if (jb[id] != NB) { map[i] = jb[id]; }
            else { map[i] = jb[id] = j++; }
        }
        else { map[i] = j++; }
    }

    orbit_list<N, element_type> ola(ctrla.req_const_symmetry());
    orbit_list<M, element_type> olb(m_sym);
    for (typename orbit_list<M, element_type>::iterator iob = olb.begin();
            iob != olb.end(); iob++) {

        index<N> idxa;
        index<M> idxb;
        olb.get_index(iob, idxb);
        idxb.permute(pinv);

        for(size_t i = 0; i < N; i++) idxa[i] = idxb[map[i]];

        orbit<N, element_type> oa(ctrla.req_const_symmetry(), idxa);
        if(! ola.contains(oa.get_acindex())) continue;

        abs_index<N> cidxa(oa.get_acindex(), bidimsa);

        if(ctrla.req_is_zero_block(cidxa.get_index())) continue;

        m_sch.insert(olb.get_abs_index(iob));
    }
}


template<size_t N, size_t M, typename Traits, typename Timed>
gen_bto_diag_task<N, M, Traits, Timed>::gen_bto_diag_task(
        gen_bto_diag<N, M, Traits, Timed> &bto,
        temp_block_tensor_type &btb,
        const index<M> &idx,
        gen_block_stream_i<M, bti_traits> &out) :

        m_bto(bto), m_btb(btb), m_idx(idx), m_out(out) {

}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_diag_task<N, M, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<M>::type
            rd_block_type;
    typedef typename bti_traits::template wr_block_type<M>::type
            wr_block_type;

    tensor_transf<M, element_type> tr0;
    gen_block_tensor_ctrl<M, bti_traits> cb(m_btb);
    {
        wr_block_type &blkb = cb.req_block(m_idx);
        m_bto.compute_block_untimed(true, m_idx, tr0, blkb);
        cb.ret_block(m_idx);
    }

    {
        rd_block_type &blkb = cb.req_const_block(m_idx);
        m_out.put(m_idx, blkb, tr0);
        cb.ret_const_block(m_idx);
    }

    cb.req_zero_block(m_idx);
}


template<size_t N, size_t M, typename Traits, typename Timed>
gen_bto_diag_task_iterator<N, M, Traits, Timed>::gen_bto_diag_task_iterator(
    gen_bto_diag<N, M, Traits, Timed> &bto,
    temp_block_tensor_type &btb,
    gen_block_stream_i<M, bti_traits> &out) :

    m_bto(bto), m_btb(btb), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, size_t M, typename Traits, typename Timed>
bool gen_bto_diag_task_iterator<N, M, Traits, Timed>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, size_t M, typename Traits, typename Timed>
libutil::task_i *gen_bto_diag_task_iterator<N, M, Traits, Timed>::get_next() {

    dimensions<M> bidims = m_btb.get_bis().get_block_index_dims();
    index<M> idx;
    abs_index<M>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    gen_bto_diag_task<N, M, Traits, Timed> *t =
        new gen_bto_diag_task<N, M, Traits, Timed>(m_bto, m_btb, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, size_t M, typename Traits>
void gen_bto_diag_task_observer<N, M, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_DIAG_IMPL_H
