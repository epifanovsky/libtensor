#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BASIC_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BASIC_IMPL_H

#include <map>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/symmetry/so_permute.h>
#include "../gen_bto_aux_copy.h"
#include "../gen_bto_copy.h"
#include "../gen_block_tensor_ctrl.h"
#include "gen_bto_contract2_block_impl.h"
#include "gen_bto_contract2_clst_builder.h"
#include "gen_bto_unfold_symmetry.h"
#include "gen_bto_contract2_batch.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_prepare_clst_task : public libutil::task_i {
private:
    gen_bto_contract2_clst_builder<N, M, K, Traits> &m_bto;

public:
    gen_bto_contract2_prepare_clst_task(
        gen_bto_contract2_clst_builder<N, M, K, Traits> &bto);

    virtual ~gen_bto_contract2_prepare_clst_task() { }
    virtual void perform();

};


template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_prepare_clst_task_iterator :
    public libutil::task_iterator_i {

private:
    std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*> &m_clstb;
    typename std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*>::iterator m_i;

public:
    gen_bto_contract2_prepare_clst_task_iterator(
        std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*> &clstb);

    virtual ~gen_bto_contract2_prepare_clst_task_iterator() { }
    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N + M>::type
        temp_block_tensor_c_type;
    typedef typename gen_bto_contract2_clst<N, M, K, element_type>::list_type
        contr_list_type;

private:
    gen_bto_contract2_block<N, M, K, Traits, Timed> &m_bto;
    const contr_list_type &m_clst;
    temp_block_tensor_c_type &m_btc;
    index<N + M> m_idxc;
    gen_block_stream_i<N + M, bti_traits> &m_out;

public:
    gen_bto_contract2_task(
        gen_bto_contract2_block<N, M, K, Traits, Timed> &bto,
        const contr_list_type &clst,
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
    const std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*> &m_clstb;
    temp_block_tensor_c_type &m_btc;
    dimensions<N + M> m_bidimsc;
    gen_block_stream_i<N + M, bti_traits> &m_out;
    const std::vector<size_t> &m_blst;
    typename std::vector<size_t>::const_iterator m_i;

public:
    gen_bto_contract2_task_iterator(
        gen_bto_contract2_block<N, M, K, Traits, Timed> &bto,
        const std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*> &clstb,
        temp_block_tensor_c_type &btc,
        const std::vector<size_t> &blst,
        gen_block_stream_i<N + M, bti_traits> &out);

    virtual ~gen_bto_contract2_task_iterator() { }
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
gen_bto_contract2_batch<N, M, K, Traits, Timed>::gen_bto_contract2_batch(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    const permutation<NA> &perma,
    const scalar_transf<element_type> &ka,
    const std::vector<size_t> &batcha,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const permutation<NB> &permb,
    const scalar_transf<element_type> &kb,
    const std::vector<size_t> &batchb,
    const block_index_space<NC> &bisc,
    const scalar_transf<element_type> &kc) :

    m_contr(contr), m_bta(bta), m_perma(perma), m_ka(ka), m_batcha(batcha),
    m_btb(btb), m_permb(permb), m_kb(kb), m_batchb(batchb),
    m_bisc(bisc), m_kc(kc) {

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2_batch<N, M, K, Traits, Timed>::perform(
    const std::vector<size_t> &blst,
    gen_block_stream_i<NC, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NA>::type
        temp_block_tensor_a_type;
    typedef typename Traits::template temp_block_tensor_type<NB>::type
        temp_block_tensor_b_type;
    typedef typename Traits::template temp_block_tensor_type<NC>::type
        temp_block_tensor_c_type;

    typedef gen_bto_copy< NA, Traits, Timed> gen_bto_copy_a_type;
    typedef gen_bto_copy< NB, Traits, Timed> gen_bto_copy_b_type;
    typedef typename gen_bto_contract2_clst<N, M, K, element_type>::list_type
        contr_list;

    gen_bto_contract2_batch::start_timer();

    try {

        out.open();

        block_index_space<NA> bisa2(m_bta.get_bis());
        bisa2.permute(m_perma);
        block_index_space<NB> bisb2(m_btb.get_bis());
        bisb2.permute(m_permb);

        dimensions<NA> bidimsa = bisa2.get_block_index_dims();
        dimensions<NB> bidimsb = bisb2.get_block_index_dims();
        dimensions<NC> bidimsc = m_bisc.get_block_index_dims();

        temp_block_tensor_a_type bta2(bisa2);
        temp_block_tensor_b_type btb2(bisb2);
        temp_block_tensor_c_type btc(m_bisc);

        symmetry<NA, element_type> syma2(bisa2);
        symmetry<NB, element_type> symb2(bisb2);

        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
            so_permute<NA, element_type>(ca.req_const_symmetry(), m_perma).
                perform(syma2);
        }
        {
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
            so_permute<NB, element_type>(cb.req_const_symmetry(), m_permb).
                perform(symb2);
        }

        {
            tensor_transf<NA, element_type> tra(m_perma);
            gen_bto_aux_copy<NA, Traits> cpaout(syma2, bta2);
            gen_bto_copy_a_type(m_bta, tra).perform(m_batcha, cpaout);

            tensor_transf<NB, element_type> trb(m_permb);
            gen_bto_aux_copy<NB, Traits> cpbout(symb2, btb2);
            gen_bto_copy_b_type(m_btb, trb).perform(m_batchb, cpbout);
        }

        std::vector<size_t> blsta, blstb;
        {
            gen_block_tensor_rd_ctrl<NA, bti_traits> ca2(bta2);
            ca2.req_nonzero_blocks(blsta);
        }
        {
            gen_block_tensor_rd_ctrl<NB, bti_traits> cb2(btb2);
            cb2.req_nonzero_blocks(blstb);
        }
        block_list<NA> bla(bidimsa, blsta);
        block_list<NB> blb(bidimsb, blstb);

        std::set<size_t> blsta2, blstb2;

        std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*> clstb;
        for(typename std::vector<size_t>::const_iterator i = blst.begin();
            i != blst.end(); ++i) {

            index<NC> idxc;
            abs_index<NC>::get_index(*i, bidimsc, idxc);
            gen_bto_contract2_clst_builder<N, M, K, Traits> *clstop =
                new gen_bto_contract2_clst_builder<N, M, K, Traits>(m_contr,
                    syma2, symb2, bla, blb, bidimsc, idxc);
            clstb.insert(std::make_pair(*i, clstop));
        }
        {
            gen_bto_contract2_prepare_clst_task_iterator<N, M, K, Traits> ti(clstb);
            gen_bto_contract2_task_observer<N, M, K> to;
            libutil::thread_pool::submit(ti, to);
        }
        for(typename std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*>::iterator i = clstb.begin();
            i != clstb.end(); ++i) {
            const contr_list &clst = i->second->get_clst();
            for(typename contr_list::const_iterator j = clst.begin();
                j != clst.end(); ++j) {
                blsta2.insert(j->get_aindex_a());
                blstb2.insert(j->get_aindex_b());
            }
        }
        blsta.clear();
        blstb.clear();
        blsta.insert(blsta.begin(), blsta2.begin(), blsta2.end());
        blsta2.clear();
        blstb.insert(blstb.begin(), blstb2.begin(), blstb2.end());
        blstb2.clear();

        gen_bto_unfold_symmetry<NA, Traits>().perform(blsta, bta2);
        gen_bto_unfold_symmetry<NB, Traits>().perform(blstb, btb2);

        gen_bto_contract2_block<N, M, K, Traits, Timed> bto(m_contr,
            m_bta, bta2, syma2, bla, m_ka, m_btb, btb2, symb2, blb, m_kb,
            m_bisc, m_kc);
        gen_bto_contract2_task_iterator<N, M, K, Traits, Timed> ti(bto,
            clstb, btc, blst, out);
        gen_bto_contract2_task_observer<N, M, K> to;
        libutil::thread_pool::submit(ti, to);

        for(typename std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*>::iterator i = clstb.begin();
            i != clstb.end(); ++i) {
            delete i->second;
            i->second = 0;
        }
        clstb.clear();

        out.close();

    } catch(...) {
        gen_bto_contract2_batch::stop_timer();
        throw;
    }

    gen_bto_contract2_batch::stop_timer();
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_prepare_clst_task<N, M, K, Traits>::
gen_bto_contract2_prepare_clst_task(
    gen_bto_contract2_clst_builder<N, M, K, Traits> &bto) :

    m_bto(bto) {

}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_prepare_clst_task<N, M, K, Traits>::perform() {

    m_bto.build_list(false);
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_prepare_clst_task_iterator<N, M, K, Traits>::
gen_bto_contract2_prepare_clst_task_iterator(
    std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*> &clstb) :

    m_clstb(clstb), m_i(m_clstb.begin()) {

}


template<size_t N, size_t M, size_t K, typename Traits>
bool gen_bto_contract2_prepare_clst_task_iterator<N, M, K, Traits>::
has_more() const {

    return m_i != m_clstb.end();
}


template<size_t N, size_t M, size_t K, typename Traits>
libutil::task_i *
gen_bto_contract2_prepare_clst_task_iterator<N, M, K, Traits>::get_next() {

    gen_bto_contract2_prepare_clst_task<N, M, K, Traits> *t =
        new gen_bto_contract2_prepare_clst_task<N, M, K, Traits>(
            *m_i->second);
    ++m_i;
    return t;
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_contract2_task<N, M, K, Traits, Timed>::gen_bto_contract2_task(
    gen_bto_contract2_block<N, M, K, Traits, Timed> &bto,
    const contr_list_type &clst,
    temp_block_tensor_c_type &btc,
    const index<N + M> &idxc,
    gen_block_stream_i<N + M, bti_traits> &out) :

    m_bto(bto), m_clst(clst), m_btc(btc), m_idxc(idxc), m_out(out) {

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
        m_bto.compute_block(m_clst, true, m_idxc, tr0, blkc);
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
    const std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*> &clstb,
    temp_block_tensor_c_type &btc,
    const std::vector<size_t> &blst,
    gen_block_stream_i<N + M, bti_traits> &out) :

    m_bto(bto), m_clstb(clstb), m_btc(btc),
    m_bidimsc(m_btc.get_bis().get_block_index_dims()),
    m_blst(blst), m_out(out), m_i(m_blst.begin()) {

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
bool gen_bto_contract2_task_iterator<N, M, K, Traits, Timed>::has_more() const {

    return m_i != m_blst.end();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
libutil::task_i *
gen_bto_contract2_task_iterator<N, M, K, Traits, Timed>::get_next() {

    abs_index<N + M> aidxc(*m_i, m_bidimsc);
    typename std::map<size_t, gen_bto_contract2_clst_builder<N, M, K, Traits>*>::const_iterator iclst =
        m_clstb.find(aidxc.get_abs_index());
    gen_bto_contract2_task<N, M, K, Traits, Timed> *t =
        new gen_bto_contract2_task<N, M, K, Traits, Timed>(m_bto,
            iclst->second->get_clst(), m_btc, aidxc.get_index(), m_out);
    ++m_i;
    return t;
}


template<size_t N, size_t M, size_t K>
void gen_bto_contract2_task_observer<N, M, K>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BASIC_IMPL_H
