#ifndef LIBTENSOR_GEN_BTO_DIRSUM_IMPL_H
#define LIBTENSOR_GEN_BTO_DIRSUM_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/core/mask.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_dirsum.h"
#include "gen_bto_dirsum_sym_impl.h"

namespace libtensor {


template<size_t N, size_t M, typename Traits, typename Timed>
class gen_bto_dirsum_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N + M>::type
        temp_block_tensor_type;

private:
    gen_bto_dirsum<N, M, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btc;
    index<N + M> m_idx;
    gen_block_stream_i<N + M, bti_traits> &m_out;

public:
    gen_bto_dirsum_task(
        gen_bto_dirsum<N, M, Traits, Timed> &bto,
        temp_block_tensor_type &btc,
        const index<N + M> &idx,
        gen_block_stream_i<N + M, bti_traits> &out);

    virtual ~gen_bto_dirsum_task() { }
    virtual void perform();
};


template<size_t N, size_t M, typename Traits, typename Timed>
class gen_bto_dirsum_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N + M>::type
        temp_block_tensor_type;

private:
    gen_bto_dirsum<N, M, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btc;
    gen_block_stream_i<N + M, bti_traits> &m_out;
    const assignment_schedule<N + M, element_type> &m_sch;
    typename assignment_schedule<N + M, element_type>::iterator m_i;

public:
    gen_bto_dirsum_task_iterator(
        gen_bto_dirsum<N, M, Traits, Timed> &bto,
        temp_block_tensor_type &btc,
        gen_block_stream_i<N + M, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, typename Traits>
class gen_bto_dirsum_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);
};


template<size_t N, size_t M, typename Traits, typename Timed>
gen_bto_dirsum<N, M, Traits, Timed>::gen_bto_dirsum(
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const scalar_transf_type &ka,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const scalar_transf_type &kb,
        const tensor_transf_type &trc) :

    m_bta(bta), m_btb(btb), m_ka(ka), m_kb(kb), m_trc(trc),
    m_symc(m_bta, m_ka, m_btb, m_kb, m_trc.get_perm()),
    m_bidimsa(m_bta.get_bis().get_block_index_dims()),
    m_bidimsb(m_btb.get_bis().get_block_index_dims()),
    m_bidimsc(m_symc.get_bis().get_block_index_dims()),
    m_sch(m_symc.get_bis().get_block_index_dims()) {

    make_schedule();
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_dirsum<N, M, Traits, Timed>::perform(
        gen_block_stream_i<NC, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NC>::type
        temp_block_tensor_type;

    gen_bto_dirsum::start_timer();

    try {

        out.open();

        temp_block_tensor_type btc(m_symc.get_bis());

        gen_bto_dirsum_task_iterator<N, M, Traits, Timed> ti(*this, btc, out);
        gen_bto_dirsum_task_observer<N, M, Traits> to;
        libutil::thread_pool::submit(ti, to);

        out.close();

    } catch(...) {
        gen_bto_dirsum::stop_timer();
        throw;
    }

    gen_bto_dirsum::stop_timer();
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_dirsum<N, M, Traits, Timed>::compute_block(
        bool zero,
        const index<NC> &idxc,
        const tensor_transf<NC, element_type> &trc,
        wr_block_type &blkc) {

    gen_bto_dirsum::start_timer();

    try {

        compute_block_untimed(zero, idxc, trc, blkc);

    } catch(...) {
        gen_bto_dirsum::stop_timer();
        throw;
    }

    gen_bto_dirsum::stop_timer();
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_dirsum<N, M, Traits, Timed>::compute_block_untimed(
        bool zero,
        const index<NC> &idxc,
        const tensor_transf_type &trc,
        wr_block_type &blkc) {

    typedef typename Traits::template to_scatter_type<NA, NB>::type
            to_scatter_a_type;
    typedef typename Traits::template to_scatter_type<NB, NA>::type
            to_scatter_b_type;
    typedef typename Traits::template to_dirsum_type<NA, NB>::type
            to_dirsum_type;
    typedef typename Traits::template to_set_type<NC>::type to_set_type;

    abs_index<NC> aic(idxc, m_bidimsc);
    typename schedule_t::const_iterator isch =
            m_op_sch.find(aic.get_abs_index());

    if(isch == m_op_sch.end()) {
        if(zero) to_set_type().perform(blkc);
        return;
    }

    const schrec &rec = isch->second;

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

    abs_index<NA> aia(rec.absidxa, m_bidimsa);
    abs_index<NB> aib(rec.absidxb, m_bidimsb);

    if(rec.zerob) {
        permutation<NC> cycc;
        for(size_t i = 0; i < NC - 1; i++) cycc.permute(i, i + 1);
        permutation<NC> permc2;
        for(size_t i = 0; i < NA; i++) permc2.permute(cycc);

        tensor_transf_type trc1(permc2);
        trc1.transform(rec.ka);
        trc1.transform(rec.trc);
        trc1.transform(trc);

        rd_block_a_type &blka = ca.req_const_block(aia.get_index());
        to_scatter_a_type(blka, trc1).perform(zero, blkc);
        ca.ret_const_block(aia.get_index());
    } else if(rec.zeroa) {
        tensor_transf_type trc1;
        trc1.transform(rec.kb);
        trc1.transform(rec.trc);
        trc1.transform(trc);

        rd_block_b_type &blkb = cb.req_const_block(aib.get_index());
        to_scatter_b_type(blkb, trc1).perform(zero, blkc);
        cb.ret_const_block(aib.get_index());
    } else {
        tensor_transf_type trc1(rec.trc);
        trc1.transform(trc);

        rd_block_a_type &blka = ca.req_const_block(aia.get_index());
        rd_block_b_type &blkb = cb.req_const_block(aib.get_index());
        to_dirsum_type(blka, rec.ka, blkb, rec.kb, trc1).perform(zero, blkc);
        ca.ret_const_block(aia.get_index());
        cb.ret_const_block(aib.get_index());
    }
}

template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_dirsum<N, M, Traits, Timed>::make_schedule() {

    gen_bto_dirsum::start_timer("make_schedule");

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta);
    gen_block_tensor_rd_ctrl<M, bti_traits> cb(m_btb);

    orbit_list<NA, element_type> ola(ca.req_const_symmetry());
    orbit_list<NB, element_type> olb(cb.req_const_symmetry());
    orbit_list<NC, element_type> olc(get_symmetry());

    for(typename orbit_list<NA, element_type>::iterator ioa = ola.begin();
            ioa != ola.end(); ioa++) {

        bool zeroa = ca.req_is_zero_block(ola.get_index(ioa));

        orbit<NA, element_type> oa(ca.req_const_symmetry(),
                ola.get_index(ioa));

        for(typename orbit_list<NB, element_type>::iterator iob = olb.begin();
                iob != olb.end(); iob++) {

            bool zerob = cb.req_is_zero_block(olb.get_index(iob));
            if(zeroa && zerob) continue;

            orbit<NB, element_type> ob(cb.req_const_symmetry(),
                    olb.get_index(iob));

            make_schedule(oa, zeroa, ob, zerob, olc);
        }
    }

    // check if there are orbits in olc which have not been added to schedule
    tensor_transf_type trc_inv(m_trc, true);
    for (typename orbit_list<NC, element_type>::iterator ioc = olc.begin();
            ioc != olc.end(); ioc++) {

        if (m_sch.contains(olc.get_abs_index(ioc))) continue;

        // identify index of A and index of B
        index<NC> idxc(olc.get_index(ioc));
        idxc.permute(trc_inv.get_perm());
        index<NA> idxa;
        for (register size_t i = 0; i < NA; i++) idxa[i] = idxc[i];
        index<NB> idxb;
        for (register size_t i = 0; i < NB; i++) idxb[i] = idxc[i + NA];

        orbit<NA, double> oa(ca.req_const_symmetry(), idxa);
        orbit<NB, double> ob(cb.req_const_symmetry(), idxb);

        bool zeroa = ! oa.is_allowed();
        bool zerob = ! ob.is_allowed();

        if (zeroa == zerob) continue;

        if (! zeroa) {
            abs_index<NA> ai(oa.get_abs_canonical_index(),
                    m_bta.get_bis().get_block_index_dims());
            zeroa = ca.req_is_zero_block(ai.get_index());
        }


        if (! zerob) {
            abs_index<NB> bi(ob.get_abs_canonical_index(),
                    m_btb.get_bis().get_block_index_dims());
            zerob = cb.req_is_zero_block(bi.get_index());
        }

        make_schedule(oa, zeroa, ob, zerob, olc);
    }

    gen_bto_dirsum::stop_timer("make_schedule");
}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_dirsum<N, M, Traits, Timed>::make_schedule(
        const orbit<NA, element_type> &oa, bool zeroa,
        const orbit<NB, element_type> &ob, bool zerob,
        const orbit_list<NC, element_type> &olc) {

    sequence<NA, size_t> seqa(0);
    sequence<NB, size_t> seqb(0);
    sequence<NC, size_t> seqc1(0), seqc2(0);

    for(register size_t i = 0; i < NC; i++) seqc1[i] = i;

    for(typename orbit<NA, element_type>::iterator ia = oa.begin();
            ia != oa.end(); ia++) {

        abs_index<NA> aidxa(oa.get_abs_index(ia), m_bidimsa);
        const index<NA> &idxa = aidxa.get_index();
        const tensor_transf<NA, element_type> &tra =
                oa.get_transf(aidxa.get_abs_index());

        for(register size_t i = 0; i < NA; i++) seqa[i] = i;
        tra.get_perm().apply(seqa);

        for(typename orbit<NB, element_type>::iterator ib = ob.begin();
                ib != ob.end(); ib++) {

            abs_index<NB> aidxb(ob.get_abs_index(ib), m_bidimsb);
            const index<NB> &idxb = aidxb.get_index();
            const tensor_transf<NB, element_type> &trb =
                    ob.get_transf(aidxb.get_abs_index());

            for(register size_t i = 0; i < NB; i++) seqb[i] = i;
            trb.get_perm().apply(seqb);

            index<NC> idxc;
            for(register size_t i = 0; i < NA; i++) {
                idxc[i] = idxa[i];
                seqc2[i] = seqa[i];
            }
            for(register size_t i = 0, j = NA; i < NB; i++, j++) {
                idxc[j] = idxb[i];
                seqc2[j] = NA + seqb[i];
            }

            idxc.permute(m_trc.get_perm());
            m_trc.get_perm().apply(seqc2);

            abs_index<NC> aidxc(idxc, m_bidimsc);
            if(! olc.contains(aidxc.get_abs_index())) continue;

            permutation_builder<NC> pbc(seqc2, seqc1);
            schrec rec;
            rec.absidxa = oa.get_abs_canonical_index();
            rec.absidxb = ob.get_abs_canonical_index();
            rec.zeroa = zeroa;
            rec.zerob = zerob;
            rec.ka.transform(tra.get_scalar_tr()).transform(m_ka);
            rec.kb.transform(trb.get_scalar_tr()).transform(m_kb);
            rec.trc.transform(tensor_transf_type(pbc.get_perm(),
                    m_trc.get_scalar_tr()));
            m_op_sch.insert(
                    std::pair<size_t, schrec>(aidxc.get_abs_index(), rec));
            m_sch.insert(aidxc.get_abs_index());
        } // for ib
    } // for ia
}


template<size_t N, size_t M, typename Traits, typename Timed>
gen_bto_dirsum_task<N, M, Traits, Timed>::gen_bto_dirsum_task(
        gen_bto_dirsum<N, M, Traits, Timed> &bto,
        temp_block_tensor_type &btc,
        const index<N + M> &idx,
        gen_block_stream_i<N + M, bti_traits> &out) :

    m_bto(bto), m_btc(btc), m_idx(idx), m_out(out) {

}


template<size_t N, size_t M, typename Traits, typename Timed>
void gen_bto_dirsum_task<N, M, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<N + M>::type
        rd_block_type;
    typedef typename bti_traits::template wr_block_type<N + M>::type
        wr_block_type;

    tensor_transf<N + M, element_type> tr0;
    gen_block_tensor_ctrl<N + M, bti_traits> cc(m_btc);

    {
        wr_block_type &blkc = cc.req_block(m_idx);
        m_bto.compute_block(true, m_idx, tr0, blkc);
        cc.ret_block(m_idx);
    }

    {
        rd_block_type &blkc = cc.req_const_block(m_idx);
        m_out.put(m_idx, blkc, tr0);
        cc.ret_const_block(m_idx);
    }

    cc.req_zero_block(m_idx);
}


template<size_t N, size_t M, typename Traits, typename Timed>
gen_bto_dirsum_task_iterator<N, M, Traits, Timed>::gen_bto_dirsum_task_iterator(
    gen_bto_dirsum<N, M, Traits, Timed> &bto,
    temp_block_tensor_type &btc,
    gen_block_stream_i<N + M, bti_traits> &out) :

    m_bto(bto), m_btc(btc), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, size_t M, typename Traits, typename Timed>
bool gen_bto_dirsum_task_iterator<N, M, Traits, Timed>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, size_t M, typename Traits, typename Timed>
libutil::task_i *gen_bto_dirsum_task_iterator<N, M, Traits, Timed>::get_next() {

    dimensions<N + M> bidims = m_btc.get_bis().get_block_index_dims();
    index<N + M> idx;
    abs_index<N + M>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    gen_bto_dirsum_task<N, M, Traits, Timed> *t =
        new gen_bto_dirsum_task<N, M, Traits, Timed>(m_bto, m_btc, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, size_t M, typename Traits>
void gen_bto_dirsum_task_observer<N, M, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor


#endif // LIBTENOSR_GEN_BTO_DIRSUM_IMPL_H
