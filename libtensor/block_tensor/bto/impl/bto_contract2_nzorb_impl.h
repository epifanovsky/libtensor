#ifndef LIBTENSOR_BTO_CONTRACT2_NZORB_IMPL_H
#define LIBTENSOR_BTO_CONTRACT2_NZORB_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include "../bto_contract2_nzorb.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_nzorb_task : public libutil::task_i {
public:
    bto_contract2_nzorb_task();
    virtual ~bto_contract2_nzorb_task() { }
    virtual void perform();

};


template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_nzorb_task_iterator : public libutil::task_iterator_i {
private:
    block_tensor_ctrl<N + K, T> &m_ca;
    block_tensor_ctrl<M + K, T> &m_cb;
    orbit_list<N + K, T> m_ola;
    typename orbit_list<N + K, T>::iterator m_ioa;

public:
    bto_contract2_nzorb_task_iterator(
        block_tensor_ctrl<N + K, T> &ca,
        block_tensor_ctrl<M + K, T> &cb);

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

    block_tensor_ctrl<N + K, T> ca(m_bta);
    block_tensor_ctrl<M + K, T> cb(m_btb);

    ca.req_sync_on();
    cb.req_sync_on();

    bto_contract2_nzorb_task_iterator<N, M, K, T> ti(ca, cb);
    bto_contract2_nzorb_task_observer<N, M, K, T> to;
    libutil::thread_pool::submit(ti, to);

    ca.req_sync_off();
    cb.req_sync_off();
}


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_nzorb_task<N, M, K, T>::bto_contract2_nzorb_task() {

}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_nzorb_task<N, M, K, T>::perform() {

}


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_nzorb_task_iterator<N, M, K, T>::
    bto_contract2_nzorb_task_iterator(block_tensor_ctrl<N + K, T> &ca,
    block_tensor_ctrl<M + K, T> &cb) :

    m_ca(ca), m_cb(cb), m_ola(ca.req_const_symmetry()), m_ioa(m_ola.begin()) {

    while(m_ioa != m_ola.end()) {
        if(!m_ca.req_is_zero_block(m_ola.get_index(m_ioa))) break;
        ++m_ioa;
    }
}


template<size_t N, size_t M, size_t K, typename T>
bool bto_contract2_nzorb_task_iterator<N, M, K, T>::has_more() const {

    return m_ioa != m_ola.end();
}


template<size_t N, size_t M, size_t K, typename T>
libutil::task_i *bto_contract2_nzorb_task_iterator<N, M, K, T>::get_next() {

    bto_contract2_nzorb_task<N, M, K, T> *t =
        new bto_contract2_nzorb_task<N, M, K, T>();

    ++m_ioa;
    while(m_ioa != m_ola.end()) {
        if(!m_ca.req_is_zero_block(m_ola.get_index(m_ioa))) break;
        ++m_ioa;
    }

    return t;
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_nzorb_task_observer<N, M, K, T>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_NZORB_IMPL_H
