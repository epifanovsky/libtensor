#ifndef LIBTENSOR_BTO_CONTRACT2_NZORB_IMPL_H
#define LIBTENSOR_BTO_CONTRACT2_NZORB_IMPL_H

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
#include <libtensor/core/block_tensor_ctrl.h>
#include "../bto_contract2_clst.h"
#include "../bto_contract2_nzorb.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_nzorb_task : public libutil::task_i {
private:
    bto_contract2_clst<N, M, K, T> m_clst;
    dimensions<N + M> m_bidimsc;
    index<N + M> m_ic;
    std::vector<size_t> &m_blst;
    libutil::mutex &m_mtx;

public:
    bto_contract2_nzorb_task(
        const contraction2<N, M, K> &contr,
        block_tensor_i<N + K, T> &bta,
        block_tensor_i<M + K, T> &btb,
        const orbit_list<N + K, T> &ola,
        const orbit_list<M + K, T> &olb,
        const dimensions<N + K> &bidimsa,
        const dimensions<M + K> &bidimsb,
        const dimensions<N + M> &bidimsc,
        const index<N + M> &ic,
        std::vector<size_t> &blst,
        libutil::mutex &mtx);

    virtual ~bto_contract2_nzorb_task() { }
    virtual void perform();

};


template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_nzorb_task_iterator : public libutil::task_iterator_i {
private:
    const contraction2<N, M, K> &m_contr;
    block_tensor_i<N + K, T> &m_bta;
    block_tensor_i<M + K, T> &m_btb;
    const symmetry<N + M, T> &m_symc;
    const orbit_list<N + K, T> &m_ola;
    const orbit_list<M + K, T> &m_olb;
    const orbit_list<N + M, T> &m_olc;
    dimensions<N + K> m_bidimsa;
    dimensions<M + K> m_bidimsb;
    dimensions<N + M> m_bidimsc;
    typename orbit_list<N + M, T>::iterator m_ioc;
    std::vector<size_t> &m_blst;
    libutil::mutex m_mtx;

public:
    bto_contract2_nzorb_task_iterator(
        const contraction2<N, M, K> &contr,
        block_tensor_i<N + K, T> &bta,
        block_tensor_i<M + K, T> &btb,
        const symmetry<N + M, T> &symc,
        const orbit_list<N + K, T> &ola,
        const orbit_list<M + K, T> &olb,
        const orbit_list<N + M, T> &olc,
        std::vector<size_t> &blst);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_nzorb_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, size_t K, typename T>
const char *bto_contract2_nzorb<N, M, K, T>::k_clazz =
    "bto_contract2_nzorb<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_nzorb<N, M, K, T>::bto_contract2_nzorb(
    const contraction2<N, M, K> &contr, block_tensor_i<N + K, T> &bta,
    block_tensor_i<M + K, T> &btb, const symmetry<N + M, T> &symc) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_symc(symc.get_bis()) {

    so_copy<N + M, T>(symc).perform(m_symc);
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_nzorb<N, M, K, T>::build() {

    bto_contract2_nzorb<N, M, K, T>::start_timer();

    try {

        block_tensor_ctrl<N + K, T> ca(m_bta);
        block_tensor_ctrl<M + K, T> cb(m_btb);

        orbit_list<N + K, T> ola(ca.req_const_symmetry());
        orbit_list<M + K, T> olb(cb.req_const_symmetry());
        orbit_list<N + M, T> olc(m_symc);

        ca.req_sync_on();
        cb.req_sync_on();

        bto_contract2_nzorb_task_iterator<N, M, K, T> ti(m_contr, m_bta, m_btb,
            m_symc, ola, olb, olc, m_blst);
        bto_contract2_nzorb_task_observer<N, M, K, T> to;
        libutil::thread_pool::submit(ti, to);

        ca.req_sync_off();
        cb.req_sync_off();

    } catch(...) {
        bto_contract2_nzorb<N, M, K, T>::stop_timer();
        throw;
    }

    bto_contract2_nzorb<N, M, K, T>::stop_timer();
}


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_nzorb_task<N, M, K, T>::bto_contract2_nzorb_task(
    const contraction2<N, M, K> &contr, block_tensor_i<N + K, T> &bta,
    block_tensor_i<M + K, T> &btb, const orbit_list<N + K, T> &ola,
    const orbit_list<M + K, T> &olb, const dimensions<N + K> &bidimsa,
    const dimensions<M + K> &bidimsb, const dimensions<N + M> &bidimsc,
    const index<N + M> &ic, std::vector<size_t> &blst, libutil::mutex &mtx) :

    m_clst(contr, bta, btb, ola, olb, bidimsa, bidimsb, bidimsc, ic),
    m_bidimsc(bidimsc), m_ic(ic), m_blst(blst), m_mtx(mtx) {

}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_nzorb_task<N, M, K, T>::perform() {

    m_clst.build_list(true);
    if(!m_clst.is_empty()) {
        libutil::auto_lock<libutil::mutex> lock(m_mtx);
        m_blst.push_back(abs_index<N + M>::get_abs_index(m_ic, m_bidimsc));
    }
}


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_nzorb_task_iterator<N, M, K, T>::
    bto_contract2_nzorb_task_iterator(const contraction2<N, M, K> &contr,
    block_tensor_i<N + K, T> &bta, block_tensor_i<M + K, T> &btb,
    const symmetry<N + M, T> &symc, const orbit_list<N + K, T> &ola,
    const orbit_list<M + K, T> &olb, const orbit_list<N + M, T> &olc,
    std::vector<size_t> &blst) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_symc(symc),
    m_bidimsa(m_bta.get_bis().get_block_index_dims()),
    m_bidimsb(m_btb.get_bis().get_block_index_dims()),
    m_bidimsc(m_symc.get_bis().get_block_index_dims()),
    m_ola(ola), m_olb(olb), m_olc(olc), m_ioc(m_olc.begin()), m_blst(blst) {

}


template<size_t N, size_t M, size_t K, typename T>
bool bto_contract2_nzorb_task_iterator<N, M, K, T>::has_more() const {

    return m_ioc != m_olc.end();
}


template<size_t N, size_t M, size_t K, typename T>
libutil::task_i *bto_contract2_nzorb_task_iterator<N, M, K, T>::get_next() {

    bto_contract2_nzorb_task<N, M, K, T> *t =
        new bto_contract2_nzorb_task<N, M, K, T>(m_contr, m_bta, m_btb,
            m_ola, m_olb, m_bidimsa, m_bidimsb, m_bidimsc,
            m_olc.get_index(m_ioc), m_blst, m_mtx);
    ++m_ioc;
    return t;
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_nzorb_task_observer<N, M, K, T>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_NZORB_IMPL_H
