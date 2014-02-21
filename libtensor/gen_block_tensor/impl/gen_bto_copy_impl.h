#ifndef LIBTENSOR_GEN_BTO_COPY_IMPL_H
#define LIBTENSOR_GEN_BTO_COPY_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/short_orbit.h>
#include <libtensor/symmetry/so_permute.h>
#include "gen_bto_copy_bis.h"
#include "gen_bto_copy_nzorb_impl.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_copy.h"

namespace libtensor {


template<size_t N, typename Traits>
class gen_bto_full_copy_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta;
    const tensor_transf<N, element_type> &m_tra;
    const symmetry<N, element_type> &m_symb;
    const dimensions<N> &m_bidimsa;
    const dimensions<N> &m_bidimsb;
    size_t m_aia;
    gen_block_stream_i<N, bti_traits> &m_out;

public:
    gen_bto_full_copy_task(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra,
        const symmetry<N, element_type> &symb,
        const dimensions<N> &bidimsa,
        const dimensions<N> &bidimsb,
        size_t aia,
        gen_block_stream_i<N, bti_traits> &out);

    virtual ~gen_bto_full_copy_task() { }
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

};


template<size_t N, typename Traits>
class gen_bto_part_copy_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta;
    const tensor_transf<N, element_type> &m_tra;
    const dimensions<N> &m_bidimsa;
    const symmetry<N, element_type> &m_symb;
    index<N> m_ib;
    gen_block_stream_i<N, bti_traits> &m_out;

public:
    gen_bto_part_copy_task(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra,
        const dimensions<N> &bidimsa,
        const symmetry<N, element_type> &symb,
        const index<N> &ib,
        gen_block_stream_i<N, bti_traits> &out);

    virtual ~gen_bto_part_copy_task() { }
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

};


template<size_t N, typename Traits>
class gen_bto_full_copy_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta;
    const tensor_transf<N, element_type> &m_tra;
    const symmetry<N, element_type> &m_symb;
    gen_block_stream_i<N, bti_traits> &m_out;
    dimensions<N> m_bidimsa;
    dimensions<N> m_bidimsb;
    gen_block_tensor_rd_ctrl<N, bti_traits> m_ca;
    std::vector<size_t> m_blsta;
    typename std::vector<size_t>::const_iterator m_ioa;

public:
    gen_bto_full_copy_task_iterator(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra,
        const symmetry<N, element_type> &symb,
        gen_block_stream_i<N, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
class gen_bto_part_copy_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta;
    const tensor_transf<N, element_type> &m_tra;
    const symmetry<N, element_type> &m_symb;
    const std::vector<size_t> &m_blst;
    gen_block_stream_i<N, bti_traits> &m_out;
    dimensions<N> m_bidimsa;
    dimensions<N> m_bidimsb;
    gen_block_tensor_rd_ctrl<N, bti_traits> m_ca;
    std::vector<size_t>::const_iterator m_i;

public:
    gen_bto_part_copy_task_iterator(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra,
        const symmetry<N, element_type> &symb,
        const std::vector<size_t> &blst,
        gen_block_stream_i<N, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
class gen_bto_copy_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, typename Traits, typename Timed>
gen_bto_copy<N, Traits, Timed>::gen_bto_copy(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf<N, element_type> &tra) :

    m_bta(bta),
    m_tra(tra),
    m_bisb(gen_bto_copy_bis<N>(m_bta.get_bis(), tra.get_perm()).get_bisb()),
    m_symb(m_bisb),
    m_schb(m_bisb.get_block_index_dims()) {

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta);
    so_permute<N, element_type>(ca.req_const_symmetry(), m_tra.get_perm()).
        perform(m_symb);
    make_schedule();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_copy<N, Traits, Timed>::perform(
    gen_block_stream_i<N, bti_traits> &out) {

    gen_bto_copy::start_timer();

    try {

        gen_bto_full_copy_task_iterator<N, Traits> ti(m_bta, m_tra, m_symb,
            out);
        gen_bto_copy_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

    } catch(...) {
        gen_bto_copy::stop_timer();
        throw;
    }

    gen_bto_copy::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_copy<N, Traits, Timed>::perform(
    const std::vector<size_t> &blst,
    gen_block_stream_i<N, bti_traits> &out) {

    gen_bto_copy::start_timer();

    try {

        gen_bto_part_copy_task_iterator<N, Traits> ti(m_bta, m_tra, m_symb,
            blst, out);
        gen_bto_copy_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

    } catch(...) {
        gen_bto_copy::stop_timer();
        throw;
    }

    gen_bto_copy::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_copy<N, Traits, Timed>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, element_type> &trb,
    wr_block_type &blkb) {

    typedef typename Traits::template to_set_type<N>::type to_set;
    typedef typename Traits::template to_copy_type<N>::type to_copy;

    gen_bto_copy::start_timer("compute_block");

    try {

        gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta);

        tensor_transf<N, element_type> trainv(m_tra, true);

        //  Corresponding index in A
        index<N> ia(ib);
        ia.permute(trainv.get_perm());

        //  Canonical index in A
        orbit<N, double> oa(ca.req_const_symmetry(), ia, false);
        const index<N> &cia = oa.get_cindex();

        //  Transformation for block from canonical A to B
        //  B = c Tr(Bc->B) Tr(A->Bc) Tr(Ac->A) Ac
        tensor_transf<N, element_type> tra(oa.get_transf(ia));
        tra.transform(m_tra).transform(trb);

        //  Compute block in B
        if(!ca.req_is_zero_block(cia)) {
            rd_block_type &blka = ca.req_const_block(cia);
            to_copy(blka, tra).perform(zero, blkb);
            ca.ret_const_block(cia);
        } else if(zero) {
            to_set().perform(blkb);
        }

    } catch(...) {
        gen_bto_copy::stop_timer("compute_block");
        throw;
    }

    gen_bto_copy::stop_timer("compute_block");
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_copy<N, Traits, Timed>::make_schedule() {

    gen_bto_copy::start_timer("make_schedule");

    try {

        gen_bto_copy_nzorb<N, Traits> nzorb(m_bta, m_tra, m_symb);
        nzorb.build();

        const block_list<N> &blstb = nzorb.get_blst();
        for(typename block_list<N>::iterator i = blstb.begin();
                i != blstb.end(); ++i) {
            m_schb.insert(blstb.get_abs_index(i));
        }

    } catch(...) {
        gen_bto_copy::stop_timer("make_schedule");
        throw;
    }

    gen_bto_copy::stop_timer("make_schedule");
}


template<size_t N, typename Traits>
gen_bto_full_copy_task<N, Traits>::gen_bto_full_copy_task(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf<N, element_type> &tra,
    const symmetry<N, element_type> &symb,
    const dimensions<N> &bidimsa,
    const dimensions<N> &bidimsb,
    size_t aia,
    gen_block_stream_i<N, bti_traits> &out) :

    m_bta(bta), m_tra(tra), m_symb(symb),
    m_bidimsa(bidimsa), m_bidimsb(bidimsb), m_aia(aia), m_out(out) {

}


template<size_t N, typename Traits>
void gen_bto_full_copy_task<N, Traits>::perform() {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    index<N> ia;
    abs_index<N>::get_index(m_aia, m_bidimsa, ia);

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta);

    rd_block_type &ba = ca.req_const_block(ia);
    if(m_tra.get_perm().is_identity()) {
        m_out.put(ia, ba, m_tra);
    } else {
        index<N> ib(ia);
        ib.permute(m_tra.get_perm());
        orbit<N, element_type> ob(m_symb, ib, false);
        abs_index<N> acib(ob.get_acindex(), m_bidimsb);
        tensor_transf<N, element_type> trb(ob.get_transf(ib));
        trb.invert();
        tensor_transf<N, element_type> tra(m_tra);
        tra.transform(trb);
        m_out.put(acib.get_index(), ba, tra);
    }
    ca.ret_const_block(ia);
}


template<size_t N, typename Traits>
gen_bto_part_copy_task<N, Traits>::gen_bto_part_copy_task(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf<N, element_type> &tra,
    const dimensions<N> &bidimsa,
    const symmetry<N, element_type> &symb,
    const index<N> &ib,
    gen_block_stream_i<N, bti_traits> &out) :

    m_bta(bta), m_tra(tra), m_bidimsa(bidimsa), m_symb(symb), m_ib(ib),
    m_out(out) {

}


template<size_t N, typename Traits>
void gen_bto_part_copy_task<N, Traits>::perform() {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta);

    if(m_tra.get_perm().is_identity()) {
        if(!ca.req_is_zero_block(m_ib)) {
            rd_block_type &ba = ca.req_const_block(m_ib);
            m_out.put(m_ib, ba, m_tra);
            ca.ret_const_block(m_ib);
        }
    } else {
        tensor_transf<N, element_type> trainv(m_tra, true);
        index<N> ia(m_ib);
        ia.permute(trainv.get_perm());
        orbit<N, element_type> oa(ca.req_const_symmetry(), ia, false);
        abs_index<N> acia(oa.get_acindex(), m_bidimsa);
        tensor_transf<N, element_type> trb(oa.get_transf(ia));
        trb.transform(m_tra);
        if(!ca.req_is_zero_block(acia.get_index())) {
            rd_block_type &ba = ca.req_const_block(acia.get_index());
            m_out.put(m_ib, ba, trb);
            ca.ret_const_block(acia.get_index());
        }
    }
}


template<size_t N, typename Traits>
gen_bto_full_copy_task_iterator<N, Traits>::gen_bto_full_copy_task_iterator(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf<N, element_type> &tra,
    const symmetry<N, element_type> &symb,
    gen_block_stream_i<N, bti_traits> &out) :

    m_bta(bta), m_tra(tra), m_symb(symb), m_out(out),
    m_bidimsa(m_bta.get_bis().get_block_index_dims()),
    m_bidimsb(m_symb.get_bis().get_block_index_dims()),
    m_ca(m_bta) {

    m_ca.req_nonzero_blocks(m_blsta);
    m_ioa = m_blsta.begin();
}


template<size_t N, typename Traits>
bool gen_bto_full_copy_task_iterator<N, Traits>::has_more() const {

    return m_ioa != m_blsta.end();
}


template<size_t N, typename Traits>
libutil::task_i *gen_bto_full_copy_task_iterator<N, Traits>::get_next() {

    gen_bto_full_copy_task<N, Traits> *t =
        new gen_bto_full_copy_task<N, Traits>(m_bta, m_tra, m_symb, m_bidimsa,
            m_bidimsb, *m_ioa, m_out);
    ++m_ioa;
    return t;
}


template<size_t N, typename Traits>
gen_bto_part_copy_task_iterator<N, Traits>::gen_bto_part_copy_task_iterator(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf<N, element_type> &tra,
    const symmetry<N, element_type> &symb,
    const std::vector<size_t> &blst,
    gen_block_stream_i<N, bti_traits> &out) :

    m_bta(bta), m_tra(tra), m_symb(symb), m_blst(blst), m_out(out),
    m_bidimsa(m_bta.get_bis().get_block_index_dims()),
    m_bidimsb(m_symb.get_bis().get_block_index_dims()),
    m_ca(m_bta), m_i(m_blst.begin()) {

}


template<size_t N, typename Traits>
bool gen_bto_part_copy_task_iterator<N, Traits>::has_more() const {

    return m_i != m_blst.end();
}


template<size_t N, typename Traits>
libutil::task_i *gen_bto_part_copy_task_iterator<N, Traits>::get_next() {

    index<N> ib;
    abs_index<N>::get_index(*m_i, m_bidimsb, ib);
    gen_bto_part_copy_task<N, Traits> *t =
        new gen_bto_part_copy_task<N, Traits>(m_bta, m_tra, m_bidimsa, m_symb,
            ib, m_out);
    ++m_i;
    return t;
}


template<size_t N, typename Traits>
void gen_bto_copy_task_observer<N, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_COPY_IMPL_H
