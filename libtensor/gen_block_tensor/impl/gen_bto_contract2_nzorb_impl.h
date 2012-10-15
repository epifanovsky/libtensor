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
    dimensions<NC> m_bidimsc;
    index<NC> m_ic;
    std::vector<size_t> &m_blst;
    libutil::mutex &m_mtx;

public:
    gen_bto_contract2_nzorb_task(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const orbit_list<NA, element_type> &ola,
        const orbit_list<NB, element_type> &olb,
        const dimensions<NA> &bidimsa,
        const dimensions<NB> &bidimsb,
        const dimensions<NC> &bidimsc,
        const index<NC> &ic,
        std::vector<size_t> &blst,
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
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta;
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb;
    const symmetry<NC, element_type> &m_symc;
    const orbit_list<NA, element_type> &m_ola;
    const orbit_list<NB, element_type> &m_olb;
    const orbit_list<NC, element_type> &m_olc;
    dimensions<NA> m_bidimsa;
    dimensions<NB> m_bidimsb;
    dimensions<NC> m_bidimsc;
    typename orbit_list<NC, element_type>::iterator m_ioc;
    std::vector<size_t> &m_blst;
    libutil::mutex m_mtx;

public:
    gen_bto_contract2_nzorb_task_iterator(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const symmetry<NC, element_type> &symc,
        const orbit_list<NA, element_type> &ola,
        const orbit_list<NB, element_type> &olb,
        const orbit_list<NC, element_type> &olc,
        std::vector<size_t> &blst);

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

    m_contr(contr), m_bta(bta), m_btb(btb), m_symc(symc.get_bis()) {

    so_copy<NC, element_type>(symc).perform(m_symc);
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2_nzorb<N, M, K, Traits, Timed>::build() {

    gen_bto_contract2_nzorb::start_timer();

    try {

        gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
        gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

        orbit_list<NA, element_type> ola(ca.req_const_symmetry());
        orbit_list<NB, element_type> olb(cb.req_const_symmetry());
        orbit_list<NC, element_type> olc(m_symc);

        gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits> ti(m_contr,
            m_bta, m_btb, m_symc, ola, olb, olc, m_blst);
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
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const orbit_list<NA, element_type> &ola,
    const orbit_list<NB, element_type> &olb,
    const dimensions<NA> &bidimsa,
    const dimensions<NB> &bidimsb,
    const dimensions<NC> &bidimsc,
    const index<NC> &ic,
    std::vector<size_t> &blst,
    libutil::mutex &mtx) :

    m_clst_bld(contr, bta, btb, ola, olb, bidimsa, bidimsb, bidimsc, ic),
    m_bidimsc(bidimsc), m_ic(ic), m_blst(blst), m_mtx(mtx) {

}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_nzorb_task<N, M, K, Traits>::perform() {

    m_clst_bld.build_list(true);
    if(!m_clst_bld.is_empty()) {
        libutil::auto_lock<libutil::mutex> lock(m_mtx);
        m_blst.push_back(abs_index<NC>::get_abs_index(m_ic, m_bidimsc));
    }
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits>::
    gen_bto_contract2_nzorb_task_iterator(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const symmetry<NC, element_type> &symc,
    const orbit_list<NA, element_type> &ola,
    const orbit_list<NB, element_type> &olb,
    const orbit_list<NC, element_type> &olc,
    std::vector<size_t> &blst) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_symc(symc),
    m_bidimsa(m_bta.get_bis().get_block_index_dims()),
    m_bidimsb(m_btb.get_bis().get_block_index_dims()),
    m_bidimsc(m_symc.get_bis().get_block_index_dims()),
    m_ola(ola), m_olb(olb), m_olc(olc), m_ioc(m_olc.begin()), m_blst(blst) {

}


template<size_t N, size_t M, size_t K, typename Traits>
bool gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits>::has_more() const {

    return m_ioc != m_olc.end();
}


template<size_t N, size_t M, size_t K, typename Traits>
libutil::task_i*
gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits>::get_next() {

    gen_bto_contract2_nzorb_task<N, M, K, Traits> *t =
        new gen_bto_contract2_nzorb_task<N, M, K, Traits>(m_contr, m_bta, m_btb,
            m_ola, m_olb, m_bidimsa, m_bidimsb, m_bidimsc,
            m_olc.get_index(m_ioc), m_blst, m_mtx);
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
