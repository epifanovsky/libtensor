#ifndef LIBTENSOR_GEN_BTO_MULT_IMPL_H
#define LIBTENSOR_GEN_BTO_MULT_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/btod/bad_block_index_space.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_merge.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_mult.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
const char *gen_bto_mult<N, Traits, Timed>::k_clazz =
        "gen_bto_mult<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
class gen_bto_mult_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

private:
    gen_bto_mult<N, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btc;
    index<N> m_idx;
    gen_block_stream_i<N, bti_traits> &m_out;

public:
    gen_bto_mult_task(
        gen_bto_mult<N, Traits, Timed> &bto,
        temp_block_tensor_type &btc,
        const index<N> &idx,
        gen_block_stream_i<N, bti_traits> &out);

    virtual ~gen_bto_mult_task() { }
    virtual void perform();

};


template<size_t N, typename Traits, typename Timed>
class gen_bto_mult_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

private:
    gen_bto_mult<N, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btc;
    gen_block_stream_i<N, bti_traits> &m_out;
    const assignment_schedule<N, element_type> &m_sch;
    typename assignment_schedule<N, element_type>::iterator m_i;

public:
    gen_bto_mult_task_iterator(
        gen_bto_mult<N, Traits, Timed> &bto,
        temp_block_tensor_type &btc,
        gen_block_stream_i<N, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
class gen_bto_mult_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, typename Traits, typename Timed>
gen_bto_mult<N, Traits, Timed>::gen_bto_mult(
    gen_block_tensor_rd_i<N, bti_traits> &bta, const tensor_transf_type &tra,
    gen_block_tensor_rd_i<N, bti_traits> &btb, const tensor_transf_type &trb,
    bool recip, const scalar_transf<element_type> &trc) :

    m_bta(bta), m_btb(btb), m_tra(tra), m_trb(trb), m_recip(recip), m_trc(trc),
    m_bisc(block_index_space<N>(m_bta.get_bis()).permute(m_tra.get_perm())),
    m_symc(m_bisc), m_sch(m_bisc.get_block_index_dims()) {

    static const char *method = "btod_mult("
            "gen_block_tensor_rd_i<N, bti_traits> &bta, "
            "const permutation<N> &pa, "
            "gen_block_tensor_rd_i<N, bti_traits> &btb, "
            "const permutation<N> &pb, "
            "bool recip, const scalar_transf<element_type> &trc)";

    block_index_space<N> bisb(m_btb.get_bis());
    bisb.permute(m_trb.get_perm());
    if(! m_bisc.equals(bisb)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "bta, btb");
    }

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(bta), cb(btb);

    sequence<N + N, size_t> seq1b, seq2b;
    for (size_t i = 0; i < N; i++) {
        seq1b[i] = i; seq2b[i] = m_tra.get_perm()[i];
    }
    for (size_t i = N, j = 0; i < N + N; i++, j++) {
        seq1b[i] = i; seq2b[i] = m_trb.get_perm()[j] + N;
    }
    permutation_builder<N + N> pbb(seq2b, seq1b);

    block_index_space_product_builder<N, N> bbx(m_bisc, m_bisc,
            permutation<N + N>());

    symmetry<N + N, element_type> symx(bbx.get_bis());
    so_dirprod<N, N, element_type>(ca.req_const_symmetry(),
            cb.req_const_symmetry(), pbb.get_perm()).perform(symx);
    mask<N + N> msk;
    sequence<N + N, size_t> seq;
    for (register size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, element_type>(symx, msk, seq).perform(m_symc);

    make_schedule();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_mult<N, Traits, Timed>::perform(
        gen_block_stream_i<N, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

    gen_bto_mult::start_timer();

    try {

        out.open();

        temp_block_tensor_type btc(m_bisc);
        gen_block_tensor_ctrl<N, bti_traits> cc(btc);

        gen_bto_mult_task_iterator<N, Traits, Timed> ti(*this, btc, out);
        gen_bto_mult_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

        out.close();

    } catch(...) {
        gen_bto_mult::stop_timer();
        throw;
    }

    gen_bto_mult::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_mult<N, Traits, Timed>::compute_block(
        bool zero,
        const index<N> &ic,
        const tensor_transf_type &trc,
        wr_block_type &blkc) {

    gen_bto_mult::start_timer("compute_block");

    try {

        compute_block_untimed(zero, ic, trc, blkc);

    } catch (...) {
        gen_bto_mult::stop_timer("compute_block");
        throw;
    }

    gen_bto_mult::stop_timer("compute_block");

}

template<size_t N, typename Traits, typename Timed>
void gen_bto_mult<N, Traits, Timed>::compute_block_untimed(
        bool zero,
        const index<N> &idxc,
        const tensor_transf_type &trc,
        wr_block_type &blkc) {

    typedef typename Traits::template to_mult_type<N>::type to_mult;
    typedef typename Traits::template to_set_type<N>::type to_set;

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrla(m_bta), ctrlb(m_btb);

    permutation<N> pinva(m_tra.get_perm(), true),
            pinvb(m_trb.get_perm(), true), pinvc(trc.get_perm(), true);

    index<N> idxa(idxc), idxb(idxc);
    idxa.permute(pinva);
    idxb.permute(pinvb);

    orbit<N, element_type> oa(ctrla.req_const_symmetry(), idxa);
    abs_index<N> cidxa(oa.get_abs_canonical_index(),
            m_bta.get_bis().get_block_index_dims());
    tensor_transf_type tra(oa.get_transf(idxa));

    orbit<N, element_type> ob(ctrlb.req_const_symmetry(), idxb);
    abs_index<N> cidxb(ob.get_abs_canonical_index(),
            m_btb.get_bis().get_block_index_dims());
    tensor_transf_type trb(ob.get_transf(idxb));

    tra.transform(m_tra);
    tra.permute(pinvc);

    trb.transform(m_trb);
    trb.permute(pinvc);

    rd_block_type &blka = ctrla.req_const_block(cidxa.get_index());
    rd_block_type &blkb = ctrlb.req_const_block(cidxb.get_index());

    scalar_transf<element_type> trc1(trc.get_scalar_tr());
    trc1.transform(m_trc);

    if(zero) to_set().perform(blkc);
    to_mult(blka, tra, blkb, trb, m_recip, trc1).perform(false, blkc);

    ctrla.ret_const_block(cidxa.get_index());
    ctrlb.ret_const_block(cidxb.get_index());
}

template<size_t N, typename Traits, typename Timed>
void gen_bto_mult<N, Traits, Timed>::make_schedule() {

    static const char *method = "make_schedule()";

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrla(m_bta), ctrlb(m_btb);

    orbit_list<N, element_type> ol(m_symc);

    for (typename orbit_list<N, element_type>::iterator iol = ol.begin();
            iol != ol.end(); iol++) {

        index<N> idx(ol.get_index(iol));
        index<N> idxa(idx), idxb(idx);
        permutation<N> pinva(m_tra.get_perm(), true),
                pinvb(m_trb.get_perm(), true);
        idxa.permute(pinva);
        idxb.permute(pinvb);

        orbit<N, element_type> oa(ctrla.req_const_symmetry(), idxa);
        if (! oa.is_allowed()) continue;
        abs_index<N> cidxa(oa.get_abs_canonical_index(),
                m_bta.get_bis().get_block_index_dims());
        bool zeroa = ctrla.req_is_zero_block(cidxa.get_index());

        orbit<N, element_type> ob(ctrlb.req_const_symmetry(), idxb);
        if (! ob.is_allowed()) {
            if (m_recip)
                throw bad_parameter(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Block not allowed in btb.");

            continue;
        }

        abs_index<N> cidxb(ob.get_abs_canonical_index(),
                m_btb.get_bis().get_block_index_dims());
        bool zerob = ctrlb.req_is_zero_block(cidxb.get_index());

        if (m_recip && zerob) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "zero in btb");
        }

        if (! zeroa && ! zerob) {
            m_sch.insert(idx);
        }
    }
}


template<size_t N, typename Traits, typename Timed>
gen_bto_mult_task<N, Traits, Timed>::gen_bto_mult_task(
        gen_bto_mult<N, Traits, Timed> &bto,
        temp_block_tensor_type &btc, const index<N> &idx,
        gen_block_stream_i<N, bti_traits> &out) :

    m_bto(bto), m_btc(btc), m_idx(idx), m_out(out) {

}


template<size_t N, typename Traits, typename Timed>
void gen_bto_mult_task<N, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<N>::type
            rd_block_type;
    typedef typename bti_traits::template wr_block_type<N>::type
            wr_block_type;

    tensor_transf<N, element_type> tr0;
    gen_block_tensor_ctrl<N, bti_traits> cc(m_btc);
    {
        wr_block_type &blkc = cc.req_block(m_idx);
        m_bto.compute_block_untimed(true, m_idx, tr0, blkc);
        cc.ret_block(m_idx);
    }

    {
        rd_block_type &blkc = cc.req_const_block(m_idx);
        m_out.put(m_idx, blkc, tr0);
        cc.ret_const_block(m_idx);
    }
}


template<size_t N, typename Traits, typename Timed>
gen_bto_mult_task_iterator<N, Traits, Timed>::gen_bto_mult_task_iterator(
        gen_bto_mult<N, Traits, Timed> &bto,
        temp_block_tensor_type &btc,
        gen_block_stream_i<N, bti_traits> &out) :

    m_bto(bto), m_btc(btc), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, typename Traits, typename Timed>
bool gen_bto_mult_task_iterator<N, Traits, Timed>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, typename Traits, typename Timed>
libutil::task_i *gen_bto_mult_task_iterator<N, Traits, Timed>::get_next() {

    dimensions<N> bidims = m_btc.get_bis().get_block_index_dims();
    index<N> idx;
    abs_index<N>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    gen_bto_mult_task<N, Traits, Timed> *t =
        new gen_bto_mult_task<N, Traits, Timed>(m_bto, m_btc, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, typename Traits>
void gen_bto_mult_task_observer<N, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_MULT_IMPL_H
