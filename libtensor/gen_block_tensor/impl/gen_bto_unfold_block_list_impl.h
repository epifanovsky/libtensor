#ifndef LIBTENSOR_GEN_BTO_UNFOLD_BLOCK_LIST_IMPL_H
#define LIBTENSOR_GEN_BTO_UNFOLD_BLOCK_LIST_IMPL_H

#include <libutil/threads/auto_lock.h>
#include <libutil/threads/mutex.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/orbit.h>
#include "gen_bto_unfold_block_list.h"

namespace libtensor {


namespace {


template<size_t N, typename Traits>
class gen_bto_unfold_block_list_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    const symmetry<N, element_type> &m_sym;
    const block_list<N> &m_blst;
    size_t m_aidx;
    block_list<N> &m_blstx;
    libutil::mutex &m_mtx;

public:
    gen_bto_unfold_block_list_task(
        const symmetry<N, element_type> &sym,
        const block_list<N> &blst,
        size_t aidx,
        block_list<N> &blstx,
        libutil::mutex &mtx) :

        m_sym(sym), m_blst(blst), m_aidx(aidx), m_blstx(blstx), m_mtx(mtx)
    { }

    virtual ~gen_bto_unfold_block_list_task() { }
    virtual void perform();

};


template<size_t N, typename Traits>
class gen_bto_unfold_block_list_task_iterator :
    public libutil::task_iterator_i {

public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    const symmetry<N, element_type> &m_sym;
    const block_list<N> &m_blst;
    typename block_list<N>::iterator m_i;
    block_list<N> &m_blstx;
    libutil::mutex m_mtx;

public:
    gen_bto_unfold_block_list_task_iterator(
        const symmetry<N, element_type> &sym,
        const block_list<N> &blst,
        block_list<N> &blstx) :

        m_sym(sym), m_blst(blst), m_i(m_blst.begin()), m_blstx(blstx)
    { }

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
class gen_bto_unfold_block_list_task_observer :
    public libutil::task_observer_i {

public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t) { delete t; }

};


} // unnamed namespace


template<size_t N, typename Traits>
void gen_bto_unfold_block_list<N, Traits>::build(block_list<N> &blstx) {

    gen_bto_unfold_block_list_task_iterator<N, Traits> ti(m_sym, m_blst, blstx);
    gen_bto_unfold_block_list_task_observer<N, Traits> to;
    libutil::thread_pool::submit(ti, to);
    blstx.sort();
}


namespace {


template<size_t N, typename Traits>
void gen_bto_unfold_block_list_task<N, Traits>::perform() {

    orbit<N, element_type> o(m_sym, m_aidx, false);

    {
        libutil::auto_lock<libutil::mutex> lock(m_mtx);
        for(typename orbit<N, element_type>::iterator j = o.begin();
            j != o.end(); ++j) m_blstx.add(o.get_abs_index(j));
    }
}


template<size_t N, typename Traits>
bool gen_bto_unfold_block_list_task_iterator<N, Traits>::has_more() const {

    return m_i != m_blst.end();
}


template<size_t N, typename Traits>
libutil::task_i*
gen_bto_unfold_block_list_task_iterator<N, Traits>::get_next() {

    gen_bto_unfold_block_list_task<N, Traits> *t =
        new gen_bto_unfold_block_list_task<N, Traits>(m_sym, m_blst,
            m_blst.get_abs_index(m_i), m_blstx, m_mtx);
    ++m_i;
    return t;
}


} // unnamed namespace


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_UNFOLD_BLOCK_LIST_IMPL_H
