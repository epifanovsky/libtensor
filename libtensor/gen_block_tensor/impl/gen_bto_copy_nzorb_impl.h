#ifndef LIBTENSOR_GEN_BTO_COPY_NZORB_IMPL_H
#define LIBTENSOR_GEN_BTO_COPY_NZORB_IMPL_H

#include <algorithm>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/symmetry.h>
#include <libtensor/symmetry/so_copy.h>
#include "gen_bto_copy_nzorb.h"

namespace libtensor {


template<size_t N, typename Traits>
gen_bto_copy_nzorb<N, Traits>::gen_bto_copy_nzorb(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf<N, element_type> &tra,
    const symmetry<N, element_type> &symb) :

    m_bta(bta), m_tra(tra), m_symb(symb.get_bis()),
    m_blstb(symb.get_bis().get_block_index_dims()){

    so_copy<N, element_type>(symb).perform(m_symb);
}


namespace {


template<size_t N, typename Traits>
class gen_bto_copy_nzorb_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    const std::vector<size_t> &m_nzorba;
    size_t m_ibegin;
    size_t m_iend;
    const dimensions<N> &m_bidimsa;
    const permutation<N> &m_perma;
    const symmetry<N, element_type> &m_symb;
    block_list<N> &m_blstb;
    libutil::spinlock &m_lock;

public:
    gen_bto_copy_nzorb_task(
        const std::vector<size_t> &nzorba, size_t ibegin, size_t iend,
        const dimensions<N> &bidimsa, const permutation<N> &perma,
        const symmetry<N, element_type> &symb, block_list<N> &blstb,
        libutil::spinlock &lock);

    virtual ~gen_bto_copy_nzorb_task() { }
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

};


template<size_t N, typename Traits>
gen_bto_copy_nzorb_task<N, Traits>::gen_bto_copy_nzorb_task(
    const std::vector<size_t> &nzorba, size_t ibegin, size_t iend,
    const dimensions<N> &bidimsa, const permutation<N> &perma,
    const symmetry<N, element_type> &symb, block_list<N> &blstb,
    libutil::spinlock &lock) :

    m_nzorba(nzorba), m_ibegin(ibegin), m_iend(iend), m_bidimsa(bidimsa),
    m_perma(perma), m_symb(symb), m_blstb(blstb), m_lock(lock) {

}


template<size_t N, typename Traits>
void gen_bto_copy_nzorb_task<N, Traits>::perform() {

    std::vector<size_t> blstb;
    blstb.reserve(m_iend - m_ibegin);

    for(size_t i = m_ibegin; i != m_iend; i++) {
        index<N> bib;
        abs_index<N>::get_index(m_nzorba[i], m_bidimsa, bib);
        bib.permute(m_perma);
        short_orbit<N, element_type> ob(m_symb, bib);
        blstb.push_back(ob.get_acindex());
    }

    {
        libutil::auto_lock<libutil::spinlock> lock(m_lock);
        for(size_t i = 0; i < blstb.size(); i++) m_blstb.add(blstb[i]);
    }
}


template<size_t N, typename Traits>
class gen_bto_copy_nzorb_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    const std::vector<size_t> &m_nzorba;
    size_t m_ibegin;
    size_t m_iend;
    const dimensions<N> &m_bidimsa;
    const permutation<N> &m_perma;
    const symmetry<N, element_type> &m_symb;
    block_list<N> &m_blstb;
    libutil::spinlock m_lock;

public:
    gen_bto_copy_nzorb_task_iterator(
        const std::vector<size_t> &nzorba, const dimensions<N> &bidimsa,
        const permutation<N> &perma, const symmetry<N, element_type> &symb,
        block_list<N> &blstb);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
gen_bto_copy_nzorb_task_iterator<N, Traits>::gen_bto_copy_nzorb_task_iterator(
    const std::vector<size_t> &nzorba, const dimensions<N> &bidimsa,
    const permutation<N> &perma, const symmetry<N, element_type> &symb,
    block_list<N> &blstb) :

    m_nzorba(nzorba), m_ibegin(0), m_iend(0), m_bidimsa(bidimsa),
    m_perma(perma), m_symb(symb), m_blstb(blstb) {

}


template<size_t N, typename Traits>
bool gen_bto_copy_nzorb_task_iterator<N, Traits>::has_more() const {

    return m_iend != m_nzorba.size();
}


template<size_t N, typename Traits>
libutil::task_i *gen_bto_copy_nzorb_task_iterator<N, Traits>::get_next() {

    const size_t batch_size = 1000;

    m_ibegin = m_iend;
    m_iend = std::min(m_iend + batch_size, m_nzorba.size());
    return new gen_bto_copy_nzorb_task<N, Traits>(m_nzorba, m_ibegin, m_iend,
        m_bidimsa, m_perma, m_symb, m_blstb, m_lock);
}


template<size_t N, typename Traits>
class gen_bto_copy_nzorb_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t) {
        delete t;
    }

};


} // unnamed namespace

template<size_t N, typename Traits>
void gen_bto_copy_nzorb<N, Traits>::build() {

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta);

    bool noperm = m_tra.get_perm().is_identity();

    std::vector<size_t> nzorba;
    ca.req_nonzero_blocks(nzorba);

    m_blstb.clear();

    if(noperm) {

        for(size_t i = 0; i < nzorba.size(); i++) m_blstb.add(nzorba[i]);

    } else {

        dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();
        gen_bto_copy_nzorb_task_iterator<N, Traits> ti(nzorba, bidimsa,
            m_tra.get_perm(), m_symb, m_blstb);
        gen_bto_copy_nzorb_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

    }

    m_blstb.sort();
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_COPY_NZORB_IMPL_H
