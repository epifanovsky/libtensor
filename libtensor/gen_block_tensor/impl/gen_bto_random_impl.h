#ifndef LIBTENSOR_GEN_BTO_RANDOM_IMPL_H
#define LIBTENSOR_GEN_BTO_RANDOM_IMPL_H

#include <list>
#include <map>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit_list.h>
#include "../gen_bto_random.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
const char gen_bto_random<N, Traits, Timed>::k_clazz[] =
    "gen_bto_random<N, Traits, Timed>";


namespace {


template<size_t N, typename Traits, typename Timed>
class gen_bto_random_block : public timings<Timed> {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef tensor_transf<N, element_type> tensor_transf_type;
    typedef std::list<tensor_transf_type> transf_list_t;
    typedef std::map<size_t, transf_list_t> transf_map_t;

private:
    gen_block_tensor_wr_i<N, bti_traits> &m_bt;
    gen_block_tensor_wr_ctrl<N, bti_traits> m_ctrl;
    dimensions<N> m_bidims;

public:
    gen_bto_random_block(
        gen_block_tensor_wr_i<N, bti_traits> &bt) :
        m_bt(bt), m_ctrl(m_bt), m_bidims(m_bt.get_bis().get_block_index_dims())
    { }

    void make_block(const index<N> &idx);

private:
    bool make_transf_map(const symmetry<N, element_type> &sym,
        const index<N> &idx, const tensor_transf_type &tr,
        transf_map_t &alltransf);

};


template<size_t N, typename Traits, typename Timed>
class gen_bto_random_task : public libutil::task_i {
private:
    gen_bto_random_block<N, Traits, Timed> &m_bto;
    index<N> m_idx;

public:
    gen_bto_random_task(
        gen_bto_random_block<N, Traits, Timed> &bto,
        const index<N> &idx) :
        m_bto(bto), m_idx(idx)
    { }

    virtual ~gen_bto_random_task() { }
    virtual void perform();

};


template<size_t N, typename Traits, typename Timed>
class gen_bto_random_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

private:
    gen_bto_random_block<N, Traits, Timed> &m_bto;
    const orbit_list<N, element_type> &m_ol;
    typename orbit_list<N, element_type>::iterator m_io;

public:
    gen_bto_random_task_iterator(
        gen_bto_random_block<N, Traits, Timed> &bto,
        const orbit_list<N, element_type> &ol) :
        m_bto(bto), m_ol(ol), m_io(m_ol.begin())
    { }

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
class gen_bto_random_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


} // unnamed namespace


template<size_t N, typename Traits, typename Timed>
void gen_bto_random<N, Traits, Timed>::perform(
    gen_block_tensor_wr_i<N, bti_traits> &bt) {

    gen_bto_random::start_timer();

    try {

        gen_block_tensor_wr_ctrl<N, bti_traits> ctrl(bt);
        orbit_list<N, element_type> ol(ctrl.req_const_symmetry());

        gen_bto_random_block<N, Traits, Timed> bto(bt);
        gen_bto_random_task_iterator<N, Traits, Timed> ti(bto, ol);
        gen_bto_random_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

    } catch(...) {
        gen_bto_random::stop_timer();
        throw;
    }

    gen_bto_random::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_random<N, Traits, Timed>::perform(
    gen_block_tensor_wr_i<N, bti_traits> &bt, const index<N> &idx) {

    gen_bto_random::start_timer();

    try {

        gen_bto_random_block<N, Traits, Timed>(bt).make_block(idx);

    } catch(...) {
        gen_bto_random::stop_timer();
        throw;
    }

    gen_bto_random::stop_timer();
}


namespace {


template<size_t N, typename Traits, typename Timed>
void gen_bto_random_block<N, Traits, Timed>::make_block(const index<N> &idx) {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

    typedef typename Traits::template to_add_type<N>::type to_add;
    typedef typename Traits::template to_copy_type<N>::type to_copy;
    typedef typename Traits::template to_random_type<N>::type to_random;
    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

    const symmetry<N, element_type> &sym = m_ctrl.req_const_symmetry();
    size_t absidx = abs_index<N>::get_abs_index(idx, m_bidims);

    to_random randop;

    tensor_transf_type tr0;
    transf_map_t transf_map;

    bool allowed = make_transf_map(sym, idx, tr0, transf_map);
    typename transf_map_t::iterator ilst = transf_map.find(absidx);
    if(!allowed || ilst == transf_map.end()) {
        m_ctrl.req_zero_block(idx);
        return;
    }

    typename transf_list_t::iterator itr = ilst->second.begin();
    if(itr == ilst->second.end()) {

        wr_block_type &blk = m_ctrl.req_block(idx);
        gen_bto_random_block::start_timer("randop");
        randop.perform(true, blk);
        gen_bto_random_block::stop_timer("randop");
        m_ctrl.ret_block(idx);

    } else {

        temp_block_tensor_type btrnd(sym.get_bis()), btsymrnd(sym.get_bis());
        gen_block_tensor_ctrl<N, bti_traits> crnd(btrnd), csymrnd(btsymrnd);

        {
            wr_block_type &rnd = crnd.req_block(idx);
            gen_bto_random_block::start_timer("randop");
            randop.perform(true, rnd);
            gen_bto_random_block::stop_timer("randop");
            crnd.ret_block(idx);
        }

        scalar_transf_sum<element_type> tottr;
        {
            rd_block_type &rnd = crnd.req_const_block(idx);
            wr_block_type &symrnd = csymrnd.req_block(idx);

            tottr.add(itr->get_scalar_tr());
            to_add symop(rnd, *itr);

            for(++itr; itr != ilst->second.end(); ++itr) {
                symop.add_op(rnd, *itr);
                tottr.add(itr->get_scalar_tr());
            }
            gen_bto_random_block::start_timer("symop");
            symop.perform(true, symrnd);
            gen_bto_random_block::stop_timer("symop");

            crnd.ret_const_block(idx);
            csymrnd.ret_block(idx);
        }
        crnd.req_zero_block(idx);

        {
            rd_block_type &symrnd = csymrnd.req_const_block(idx);
            wr_block_type &blk = m_ctrl.req_block(idx);

            scalar_transf<element_type> str(tottr.get_transf());
            if(str.is_zero()) str = scalar_transf<element_type>();
            else str.invert();
            tensor_transf<N, element_type> tr(permutation<N>(), str);

            gen_bto_random_block::start_timer("copy");
            to_copy(symrnd, tr).perform(true, blk);
            gen_bto_random_block::stop_timer("copy");

            csymrnd.ret_const_block(idx);
            m_ctrl.ret_block(idx);
        }

        csymrnd.req_zero_block(idx);
    }
}


template<size_t N, typename Traits, typename Timed>
bool gen_bto_random_block<N, Traits, Timed>::make_transf_map(
    const symmetry<N, element_type> &sym, const index<N> &idx,
    const tensor_transf_type &tr, transf_map_t &alltransf) {

    size_t absidx = abs_index<N>::get_abs_index(idx, m_bidims);
    typename transf_map_t::iterator ilst = alltransf.find(absidx);
    if(ilst == alltransf.end()) {
        ilst = alltransf.insert(std::make_pair(absidx, transf_list_t())).first;
    }
    typename transf_list_t::iterator itr = ilst->second.begin();
    bool done = false;
    for(; itr != ilst->second.end(); ++itr) {
        if(*itr == tr) {
            done = true;
            break;
        }
    }
    if(done) return true;
    ilst->second.push_back(tr);

    bool allowed = true;
    for(typename symmetry<N, element_type>::iterator iset = sym.begin();
        iset != sym.end(); ++iset) {

        const symmetry_element_set<N, element_type> &eset =
            sym.get_subset(iset);
        for(typename symmetry_element_set<N, element_type>::const_iterator
            ielem = eset.begin(); ielem != eset.end(); ++ielem) {

            const symmetry_element_i<N, element_type> &elem =
                eset.get_elem(ielem);
            index<N> idx2(idx);
            tensor_transf_type tr2(tr);
            if(elem.is_allowed(idx2)) {
                elem.apply(idx2, tr2);
                allowed = make_transf_map(sym, idx2, tr2, alltransf);
            } else {
                allowed = false;
            }
        }
    }
    return allowed;
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_random_task<N, Traits, Timed>::perform() {

    m_bto.make_block(m_idx);
}


template<size_t N, typename Traits, typename Timed>
bool gen_bto_random_task_iterator<N, Traits, Timed>::has_more() const {

    return m_io != m_ol.end();
}


template<size_t N, typename Traits, typename Timed>
libutil::task_i *gen_bto_random_task_iterator<N, Traits, Timed>::get_next() {

    index<N> idx;
    m_ol.get_index(m_io, idx);
    ++m_io;
    return new gen_bto_random_task<N, Traits, Timed>(m_bto, idx);
}


template<size_t N, typename Traits>
void gen_bto_random_task_observer<N, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // unnamed namespace


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_RANDOM_IMPL_H
