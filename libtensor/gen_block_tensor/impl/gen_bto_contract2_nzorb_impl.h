#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_NZORB_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_NZORB_IMPL_H

#include <list>
#include <set>
#include <utility>
#include <vector>
#include <libutil/threads/auto_lock.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_copy.h>
#include "../gen_block_tensor_ctrl.h"
#include "gen_bto_contract2_clst_builder_impl.h"
#include "gen_bto_contract2_nzorb.h"
#include "gen_bto_unfold_block_list.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_nzorb_task : public libutil::task_i {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_contract2_clst_builder<N, M, K, Traits> m_clst_bld;
    index<NC> m_ic;
    block_list<NC> &m_blstc;
    libutil::mutex &m_mtx;

public:
    gen_bto_contract2_nzorb_task(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb,
        const block_list<NA> &blsta,
        const block_list<NB> &blstb,
        const index<NC> &ic,
        block_list<NC> &blstc,
        libutil::mutex &mtx);

    virtual ~gen_bto_contract2_nzorb_task() { }
    virtual void perform();

};


template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_nzorb_task_iterator : public libutil::task_iterator_i {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    const contraction2<N, M, K> &m_contr;
    const symmetry<NA, element_type> &m_syma;
    const symmetry<NB, element_type> &m_symb;
    const block_list<NA> &m_blsta;
    const block_list<NB> &m_blstb;
    const orbit_list<NC, element_type> &m_olc;
    typename orbit_list<NC, element_type>::iterator m_ioc;
    block_list<NC> &m_blstc;
    libutil::mutex m_mtx;

public:
    gen_bto_contract2_nzorb_task_iterator(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb,
        const block_list<NA> &blsta,
        const block_list<NB> &blstb,
        const orbit_list<NC, element_type> &olc,
        block_list<NC> &blstc);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, size_t K>
class gen_bto_contract2_nzorb_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_nzorb<N, M, K, Traits, Timed>::gen_bto_contract2_nzorb(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const symmetry<NC, element_type> &symc) :

    m_contr(contr),
    m_syma(bta.get_bis()), m_symb(btb.get_bis()), m_symc(symc.get_bis()),
    m_blsta(bta.get_bis().get_block_index_dims()),
    m_blstb(btb.get_bis().get_block_index_dims()),
    m_blstc(symc.get_bis().get_block_index_dims()) {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(btb);

    so_copy<NA, element_type>(ca.req_const_symmetry()).perform(m_syma);
    so_copy<NB, element_type>(cb.req_const_symmetry()).perform(m_symb);
    so_copy<NC, element_type>(symc).perform(m_symc);

    std::vector<size_t> blst;
    ca.req_nonzero_blocks(blst);
    for(size_t i = 0; i < blst.size(); i++) m_blsta.add(blst[i]);
    cb.req_nonzero_blocks(blst);
    for(size_t i = 0; i < blst.size(); i++) m_blstb.add(blst[i]);
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_nzorb<N, M, K, Traits, Timed>::gen_bto_contract2_nzorb(
    const contraction2<N, M, K> &contr,
    const symmetry<NA, element_type> &syma,
    const assignment_schedule<NA, element_type> &scha,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const symmetry<NC, element_type> &symc) :

    m_contr(contr),
    m_syma(syma.get_bis()), m_symb(btb.get_bis()), m_symc(symc.get_bis()),
    m_blsta(syma.get_bis().get_block_index_dims()),
    m_blstb(btb.get_bis().get_block_index_dims()),
    m_blstc(symc.get_bis().get_block_index_dims()) {

    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(btb);

    so_copy<NA, element_type>(syma).perform(m_syma);
    so_copy<NB, element_type>(cb.req_const_symmetry()).perform(m_symb);
    so_copy<NC, element_type>(symc).perform(m_symc);

    for(typename assignment_schedule<NA, element_type>::iterator ia =
        scha.begin(); ia != scha.end(); ++ia) {
        m_blsta.add(scha.get_abs_index(ia));
    }

    orbit_list<NB, element_type> olb(m_symb);
    for(typename orbit_list<NB, element_type>::iterator iol = olb.begin();
        iol != olb.end(); ++iol) {
        index<NB> idx;
        olb.get_index(iol, idx);
        if(cb.req_is_zero_block(idx)) continue;
        m_blstb.add(olb.get_abs_index(iol));
    }
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_nzorb<N, M, K, Traits, Timed>::gen_bto_contract2_nzorb(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    const symmetry<NB, element_type> &symb,
    const assignment_schedule<NB, element_type> &schb,
    const symmetry<NC, element_type> &symc) :

    m_contr(contr),
    m_syma(bta.get_bis()), m_symb(symb.get_bis()), m_symc(symc.get_bis()),
    m_blsta(bta.get_bis().get_block_index_dims()),
    m_blstb(symb.get_bis().get_block_index_dims()),
    m_blstc(symc.get_bis().get_block_index_dims()) {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(bta);

    so_copy<NA, element_type>(ca.req_const_symmetry()).perform(m_syma);
    so_copy<NB, element_type>(symb).perform(m_symb);
    so_copy<NC, element_type>(symc).perform(m_symc);

    orbit_list<NA, element_type> ola(m_syma);
    for(typename orbit_list<NA, element_type>::iterator iol = ola.begin();
        iol != ola.end(); ++iol) {
        index<NA> idx;
        ola.get_index(iol, idx);
        if(ca.req_is_zero_block(idx)) continue;
        m_blsta.add(ola.get_abs_index(iol));
    }

    for(typename assignment_schedule<NB, element_type>::iterator isch =
        schb.begin(); isch != schb.end(); ++isch) {
        m_blstb.add(schb.get_abs_index(isch));
    }
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_nzorb<N, M, K, Traits, Timed>::gen_bto_contract2_nzorb(
    const contraction2<N, M, K> &contr,
    const symmetry<NA, element_type> &syma,
    const assignment_schedule<NA, element_type> &scha,
    const symmetry<NB, element_type> &symb,
    const assignment_schedule<NB, element_type> &schb,
    const symmetry<NC, element_type> &symc) :

    m_contr(contr),
    m_syma(syma.get_bis()), m_symb(symb.get_bis()), m_symc(symc.get_bis()),
    m_blsta(syma.get_bis().get_block_index_dims()),
    m_blstb(symb.get_bis().get_block_index_dims()),
    m_blstc(symc.get_bis().get_block_index_dims()) {

    so_copy<NA, element_type>(syma).perform(m_syma);
    so_copy<NB, element_type>(symb).perform(m_symb);
    so_copy<NC, element_type>(symc).perform(m_symc);

    for(typename assignment_schedule<NA, element_type>::iterator isch =
        scha.begin(); isch != scha.end(); ++isch) {
        m_blsta.add(scha.get_abs_index(isch));
    }

    for(typename assignment_schedule<NB, element_type>::iterator isch =
        schb.begin(); isch != schb.end(); ++isch) {
        m_blstb.add(schb.get_abs_index(isch));
    }
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2_nzorb<N, M, K, Traits, Timed>::build() {

    gen_bto_contract2_nzorb::start_timer();

    try {

        block_list<NA> blstax(m_syma.get_bis().get_block_index_dims());
        block_list<NB> blstbx(m_symb.get_bis().get_block_index_dims());

        gen_bto_unfold_block_list<NA, Traits>(m_syma, m_blsta).build(blstax);
        gen_bto_unfold_block_list<NB, Traits>(m_symb, m_blstb).build(blstbx);

        orbit_list<NC, element_type> olc(m_symc);

        gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits> ti(m_contr,
            m_syma, m_symb, blstax, blstbx, olc, m_blstc);
        gen_bto_contract2_nzorb_task_observer<N, M, K> to;
        libutil::thread_pool::submit(ti, to);

    } catch(...) {
        gen_bto_contract2_nzorb::stop_timer();
        throw;
    }

    gen_bto_contract2_nzorb::stop_timer();
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_nzorb_task<N, M, K, Traits>::gen_bto_contract2_nzorb_task(
    const contraction2<N, M, K> &contr,
    const symmetry<NA, element_type> &syma,
    const symmetry<NB, element_type> &symb,
    const block_list<NA> &blsta,
    const block_list<NB> &blstb,
    const index<NC> &ic,
    block_list<NC> &blstc,
    libutil::mutex &mtx) :

    m_clst_bld(contr, syma, symb, blsta, blstb, blstc.get_dims(), ic),
    m_ic(ic), m_blstc(blstc), m_mtx(mtx) {

}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_nzorb_task<N, M, K, Traits>::perform() {

    m_clst_bld.build_list(true);
    if(! m_clst_bld.is_empty()) {
        libutil::auto_lock<libutil::mutex> lock(m_mtx);
        m_blstc.add(m_ic);
    }
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits>::
gen_bto_contract2_nzorb_task_iterator(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb,
        const block_list<NA> &blsta,
        const block_list<NB> &blstb,
        const orbit_list<NC, element_type> &olc,
        block_list<NC> &blstc) :

    m_contr(contr), m_syma(syma), m_symb(symb), m_blsta(blsta), m_blstb(blstb),
    m_olc(olc), m_ioc(m_olc.begin()), m_blstc(blstc) {

}


template<size_t N, size_t M, size_t K, typename Traits>
bool gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits>::has_more() const {

    return m_ioc != m_olc.end();
}


template<size_t N, size_t M, size_t K, typename Traits>
libutil::task_i*
gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits>::get_next() {

    index<N + M> idxc;
    m_olc.get_index(m_ioc, idxc);
    gen_bto_contract2_nzorb_task<N, M, K, Traits> *t =
        new gen_bto_contract2_nzorb_task<N, M, K, Traits>(m_contr,
            m_syma, m_symb, m_blsta, m_blstb, idxc, m_blstc, m_mtx);
    ++m_ioc;
    return t;
}


template<size_t N, size_t M, size_t K>
void gen_bto_contract2_nzorb_task_observer<N, M, K>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_NZORB_IMPL_H
