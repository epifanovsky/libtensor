#ifndef LIBTENSOR_GEN_BTO_EXTRACT_IMPL_H
#define LIBTENSOR_GEN_BTO_EXTRACT_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_reduce.h>
#include <libtensor/symmetry/so_permute.h>
#include "../gen_bto_extract.h"

namespace libtensor {


template<size_t N, size_t M, typename Traits, typename Timed>
const char *gen_bto_extract<N, M, Traits, Timed>::k_clazz =
        "gen_bto_extract<N, M, Traits, Timed>";


template<size_t N, size_t M, typename Traits, typename Timed>
class gen_bto_extract_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N - M>::type
        temp_block_tensor_type;

private:
    gen_bto_extract<N, M, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btb;
    index<N - M> m_idx;
    gen_block_stream_i<N - M, bti_traits> &m_out;

public:
    gen_bto_extract_task(
        gen_bto_extract<N, M, Traits, Timed> &bto,
        temp_block_tensor_type &btc,
        const index<N - M> &idx,
        gen_block_stream_i<N - M, bti_traits> &out);

    virtual ~gen_bto_extract_task() { }
    virtual void perform();

};


template<size_t N, size_t M, typename Traits, typename Timed>
class gen_bto_extract_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N - M>::type
        temp_block_tensor_type;

private:
    gen_bto_extract<N, M, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btb;
    gen_block_stream_i<N - M, bti_traits> &m_out;
    const assignment_schedule<N - M, double> &m_sch;
    typename assignment_schedule<N - M, double>::iterator m_i;

public:
    gen_bto_extract_task_iterator(
        gen_bto_extract<N, M, Traits, Timed> &bto,
        temp_block_tensor_type &btc,
        gen_block_stream_i<N - M, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, typename Traits>
class gen_bto_extract_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, typename Traits, typename Timed>
gen_bto_extract<N, M, Traits, Timed>::gen_bto_extract(
        gen_block_tensor_rd_i<NA, bti_traits> &bta, const mask<NA> &m,
        const index<NA> &idxbl, const index<NA> &idxibl,
        const tensor_transf_type &tr) :

    m_bta(bta), m_msk(m), m_idxbl(idxbl), m_idxibl(idxibl), m_tr(tr),
    m_bis(mk_bis(bta.get_bis(), m_msk, m_tr.get_perm())), m_sym(m_bis),
    m_sch(m_bis.get_block_index_dims()) {

    permutation<NB> pinv(m_tr.get_perm(), true);
    block_index_space<NB> bisinv(m_bis);
    bisinv.permute(pinv);

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(bta);
    symmetry<NB, element_type> sym(bisinv);

    sequence<NA, size_t> seq(0);
    mask<NA> invmsk;
    for (register size_t i = 0, j = 0; i < NA; i++) {
        invmsk[i] = !m_msk[i];
        if (invmsk[i]) seq[i] = j++;
    }

    so_reduce<N, M, element_type>(ca.req_const_symmetry(),
            invmsk, seq, index_range<NA>(idxbl, idxbl),
            index_range<NA>(idxibl, idxibl)).perform(sym);
    so_permute<NB, element_type>(sym, m_tr.get_perm()).perform(m_sym);

    make_schedule();
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_extract<N, M, Traits, Timed>::perform(
        gen_block_stream_i<NB, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NB>::type
        temp_block_tensor_type;

    gen_bto_extract::start_timer();

    try {

        temp_block_tensor_type btb(m_bis);

        gen_bto_extract_task_iterator<N, M, Traits, Timed> ti(*this, btb, out);
        gen_bto_extract_task_observer<N, M, Traits> to;
        libutil::thread_pool::submit(ti, to);

    } catch(...) {
        gen_bto_extract::stop_timer();
        throw;
    }

    gen_bto_extract::stop_timer();
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_extract<N, M, Traits, Timed>::compute_block(
        bool zero, const index<NB> &idxb,
        const tensor_transf<NB, element_type> &trb, wr_block_type &blkb) {

    gen_bto_extract::start_timer();

    try {

        compute_block_untimed(zero, idxb, trb, blkb);

    } catch(...) {
        gen_bto_extract::stop_timer();
        throw;
    }

    gen_bto_extract::stop_timer();
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_extract<N, M, Traits, Timed>::compute_block_untimed(
        bool zero, const index<NB> &idxb,
        const tensor_transf<NB, element_type> &trb, wr_block_type &blkb) {

    typedef typename Traits::template to_extract_type<N, M>::type
            to_extract_type;
    typedef typename Traits::template to_set_type<NB>::type to_set_type;

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);

    permutation<NB> pinv(m_tr.get_perm(), true);

    index<NA> idxa;
    index<NB> idxb2(idxb);
    idxb2.permute(pinv);

    for(size_t i = 0, j = 0; i < NA; i++) {
        if(m_msk[i]) {
            idxa[i] = idxb2[j++];
        } else {
            idxa[i] = m_idxbl[i];
        }
    }

    orbit<NA, element_type> oa(ca.req_const_symmetry(), idxa);

    abs_index<NA> cidxa(oa.get_acindex(),
            m_bta.get_bis().get_block_index_dims());
    tensor_transf<NA, element_type> tra(oa.get_transf(idxa));
    tra.invert();

    mask<NA> msk1(m_msk), msk2(m_msk);
    msk2.permute(tra.get_perm());

    sequence<NA, size_t> seqa1(0), seqa2(0);
    sequence<NB, size_t> seqb1(0), seqb2(0);
    for(register size_t i = 0; i < NA; i++) seqa2[i] = seqa1[i] = i;
    tra.get_perm().apply(seqa2);
    for(register size_t i = 0, j1 = 0, j2 = 0; i < NA; i++) {
        if(msk1[i]) seqb1[j1++] = seqa1[i];
        if(msk2[i]) seqb2[j2++] = seqa2[i];
    }

    permutation_builder<NB> pb(seqb2, seqb1);
    tensor_transf_type tr(pb.get_perm(), tra.get_scalar_tr());
    tr.transform(m_tr);
    tr.transform(trb);

    index<NA> idxibl2(m_idxibl);
    idxibl2.permute(tra.get_perm());

    bool zeroa = !oa.is_allowed();
    if(!zeroa) zeroa = ca.req_is_zero_block(cidxa.get_index());

    if(!zeroa) {

        rd_block_type &blka = ca.req_const_block(cidxa.get_index());

        to_extract_type(blka, msk2, idxibl2, tr).perform(zero, blkb);

        ca.ret_const_block(cidxa.get_index());

    } else if(zero) {

        to_set_type().perform(blkb);
    }
}


template<size_t N, size_t M, typename Traits, typename Timed>
block_index_space<N - M> gen_bto_extract<N, M, Traits, Timed>::mk_bis(
    const block_index_space<NA> &bis, const mask<NA> &msk,
    const permutation<NB> &perm) {

    static const char *method = "mk_bis(const block_index_space<N>&, "
        "const mask<N>&, const permutation<N - M>&)";

    dimensions<NA> idims(bis.get_dims());

    //  Compute output dimensions
    //

    index<NB> i1, i2;

    size_t m = 0, j = 0;
    size_t map[NB];//map between B and A

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
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "m");
    }

    block_index_space<NB> obis(dimensions<NB>(index_range<NB>(i1, i2)));

    mask<NB> msk_done;
    bool done = false;
    while(!done) {

        size_t i = 0;
        while(i < NB && msk_done[i]) i++;
        if(i == NB) {
            done = true;
            continue;
        }
        size_t typ = bis.get_type(map[i]);
        const split_points &splits = bis.get_splits(typ);
        mask<NB> msk_typ;
        for(size_t k = 0; k < NB; k++) {
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


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_extract<N, M, Traits, Timed>::make_schedule() {

    gen_bto_extract::start_timer("make_schedule");

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
    dimensions<NA> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<NB> pinv(m_tr.get_perm(), true);

    orbit_list<NB, double> olb(m_sym);
    for (typename orbit_list<NB, double>::iterator iob = olb.begin();
            iob != olb.end(); iob++) {

        index<NA> idxa;
        index<NB> idxb;

        olb.get_index(iob, idxb);
        idxb.permute(pinv);

        for(size_t i = 0, j = 0; i < NA; i++) {
            if(m_msk[i]) idxa[i] = idxb[j++];
            else idxa[i] = m_idxbl[i];
        }

        orbit<NA, element_type> oa(ca.req_const_symmetry(), idxa);

        abs_index<NA> cidxa(oa.get_acindex(),
                m_bta.get_bis().get_block_index_dims());

        if(!oa.is_allowed()) continue;
        if(ca.req_is_zero_block(cidxa.get_index())) continue;

        m_sch.insert(olb.get_abs_index(iob));
    }

    gen_bto_extract::stop_timer("make_schedule");

}


template<size_t N, size_t M, typename Traits, typename Timed>
gen_bto_extract_task<N, M, Traits, Timed>::gen_bto_extract_task(
        gen_bto_extract<N, M, Traits, Timed> &bto,
        temp_block_tensor_type &btb,
        const index<N - M> &idx,
        gen_block_stream_i<N - M, bti_traits> &out) :

    m_bto(bto), m_btb(btb), m_idx(idx), m_out(out) {

}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_extract_task<N, M, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<N - M>::type
        rd_block_type;
    typedef typename bti_traits::template wr_block_type<N - M>::type
        wr_block_type;

    tensor_transf<N - M, element_type> tr0;
    gen_block_tensor_ctrl<N - M, bti_traits> cb(m_btb);
    {
        wr_block_type &blk = cb.req_block(m_idx);
        m_bto.compute_block(true, m_idx, tr0, blk);
        cb.ret_block(m_idx);
    }

    {
        rd_block_type &blk = cb.req_const_block(m_idx);
        m_out.put(m_idx, blk, tr0);
        cb.ret_const_block(m_idx);
    }
    cb.req_zero_block(m_idx);
}


template<size_t N, size_t M, typename Traits, typename Timed>
gen_bto_extract_task_iterator<N, M, Traits, Timed>::gen_bto_extract_task_iterator(
    gen_bto_extract<N, M, Traits, Timed> &bto,
    temp_block_tensor_type &btb,
    gen_block_stream_i<N - M, bti_traits> &out) :

    m_bto(bto), m_btb(btb), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, size_t M, typename Traits, typename Timed>
bool gen_bto_extract_task_iterator<N, M, Traits, Timed>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, size_t M, typename Traits, typename Timed>
libutil::task_i *gen_bto_extract_task_iterator<N, M, Traits, Timed>::get_next() {

    dimensions<N - M> bidims = m_btb.get_bis().get_block_index_dims();
    index<N - M> idx;
    abs_index<N - M>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    gen_bto_extract_task<N, M, Traits, Timed> *t =
        new gen_bto_extract_task<N, M, Traits, Timed>(m_bto, m_btb, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, size_t M, typename Traits>
void gen_bto_extract_task_observer<N, M, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXTRACT_IMPL_H
