#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include "gen_bto_contract2_block_impl.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N + M>::type
        temp_block_tensor_c_type;

private:
    gen_bto_contract2_block<N, M, K, Traits, Timed> &m_bto;
    temp_block_tensor_c_type &m_btc;
    index<N + M> m_idxc;
    gen_block_stream_i<N + M, bti_traits> &m_out;

public:
    gen_bto_contract2_task(
        gen_bto_contract2_block<N, M, K, Traits, Timed> &bto,
        temp_block_tensor_c_type &btc,
        const index<N + M> &idxc,
        gen_block_stream_i<N + M, bti_traits> &out);

    virtual ~gen_bto_contract2_task() { }
    virtual void perform();

};


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N + M>::type
        temp_block_tensor_c_type;

private:
    gen_bto_contract2_block<N, M, K, Traits, Timed> &m_bto;
    temp_block_tensor_c_type &m_btc;
    dimensions<N + M> m_bidimsc;
    gen_block_stream_i<N + M, bti_traits> &m_out;
    const std::vector<size_t> &m_blst;
    typename std::vector<size_t>::const_iterator m_i;

public:
    gen_bto_contract2_task_iterator(
        gen_bto_contract2_block<N, M, K, Traits, Timed> &bto,
        temp_block_tensor_c_type &btc,
        const std::vector<size_t> &blst,
        gen_block_stream_i<N + M, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, size_t K>
class gen_bto_contract2_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2<N, M, K, Traits, Timed>::gen_bto_contract2(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_symc(m_contr, m_bta, m_btb) {

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2<N, M, K, Traits, Timed>::perform(
    const std::vector<size_t> &blst,
    gen_block_stream_i<NC, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NC>::type
        temp_block_tensor_c_type;

    gen_bto_contract2::start_timer();

    try {

        out.open();

        temp_block_tensor_c_type btc(m_symc.get_bisc());

        gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
        gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

        const symmetry<NA, element_type> &syma = ca.req_const_symmetry();
        const symmetry<NB, element_type> &symb = cb.req_const_symmetry();

        gen_bto_contract2_block<N, M, K, Traits, Timed> bto(m_contr, m_bta,
            syma, m_btb, symb, m_symc.get_bisc());
        gen_bto_contract2_task_iterator<N, M, K, Traits, Timed> ti(bto, btc,
            blst, out);
        gen_bto_contract2_task_observer<N, M, K> to;
        libutil::thread_pool::submit(ti, to);

        out.close();

    } catch(...) {
        gen_bto_contract2::stop_timer();
        throw;
    }

    gen_bto_contract2::stop_timer();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_task<N, M, K, Traits, Timed>::gen_bto_contract2_task(
    gen_bto_contract2_block<N, M, K, Traits, Timed> &bto,
    temp_block_tensor_c_type &btc,
    const index<N + M> &idxc,
    gen_block_stream_i<N + M, bti_traits> &out) :

    m_bto(bto), m_btc(btc), m_idxc(idxc), m_out(out) {

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2_task<N, M, K, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<N + M>::type
        rd_block_type;
    typedef typename bti_traits::template wr_block_type<N + M>::type
        wr_block_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;

    tensor_transf<N + M, element_type> tr0;
    gen_block_tensor_ctrl<N + M, bti_traits> cc(m_btc);

    {
        wr_block_type &blkc = cc.req_block(m_idxc);
        m_bto.compute_block(true, m_idxc, tr0, blkc);
        cc.ret_block(m_idxc);
    }

    {
        rd_block_type &blkc = cc.req_const_block(m_idxc);
        m_out.put(m_idxc, blkc, tr0);
        cc.ret_const_block(m_idxc);
    }

    cc.req_zero_block(m_idxc);
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_task_iterator<N, M, K, Traits, Timed>::
gen_bto_contract2_task_iterator(
    gen_bto_contract2_block<N, M, K, Traits, Timed> &bto,
    temp_block_tensor_c_type &btc,
    const std::vector<size_t> &blst,
    gen_block_stream_i<N + M, bti_traits> &out) :

    m_bto(bto), m_btc(btc), m_bidimsc(m_btc.get_bis().get_block_index_dims()),
    m_blst(blst), m_out(out), m_i(m_blst.begin()) {

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
bool gen_bto_contract2_task_iterator<N, M, K, Traits, Timed>::has_more() const {

    return m_i != m_blst.end();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
libutil::task_i *gen_bto_contract2_task_iterator<N, M, K, Traits, Timed>::
get_next() {

    abs_index<N + M> aidxc(*m_i, m_bidimsc);
    gen_bto_contract2_task<N, M, K, Traits, Timed> *t =
        new gen_bto_contract2_task<N, M, K, Traits, Timed>(m_bto,
            m_btc, aidxc.get_index(), m_out);
    ++m_i;
    return t;
}


template<size_t N, size_t M, size_t K>
void gen_bto_contract2_task_observer<N, M, K>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H
