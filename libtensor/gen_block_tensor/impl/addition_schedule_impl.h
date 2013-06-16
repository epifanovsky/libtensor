#ifndef LIBTENSOR_ADDITION_SCHEDULE_IMPL_H
#define LIBTENSOR_ADDITION_SCHEDULE_IMPL_H

#include <algorithm>
#include <cstring>
#include <memory>
#include <libutil/threads/auto_lock.h>
#include <libutil/threads/spinlock.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/combined_orbits.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/short_orbit.h>
#include <libtensor/core/subgroup_orbits.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include "../addition_schedule.h"

namespace libtensor {


template<size_t N, typename Traits>
const char *addition_schedule<N, Traits>::k_clazz =
    "addition_schedule<N, Traits>";


template<size_t N, typename Traits>
addition_schedule<N, Traits>::addition_schedule(
    const symmetry_type &syma,
    const symmetry_type &symb) :

    m_syma(syma), m_symb(symb), m_symc(m_symb.get_bis()) {

    permutation<N + N> perm0;
    block_index_space_product_builder<N, N> bbx(m_syma.get_bis(),
        m_symb.get_bis(), perm0);
    symmetry<N + N, element_type> symx(bbx.get_bis());
    so_dirsum<N, N, element_type>(m_syma, m_symb, perm0).perform(symx);

    mask<N + N> msk;
    sequence<N + N, size_t> seq(0);
    for (register size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, element_type>(symx, msk, seq).perform(m_symc);
}


template<size_t N, typename Traits>
addition_schedule<N, Traits>::~addition_schedule() {

    clear_schedule();
}


namespace {


template<size_t N, typename Traits>
class addition_schedule_task_1 : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename addition_schedule<N, Traits>::book_node book_node;

private:
    std::vector<size_t> m_nzorb;
    const symmetry<N, element_type> &m_syma;
    const symmetry<N, element_type> &m_symb;
    std::map<size_t, book_node> &m_booka;
    libutil::spinlock &m_lock;

public:
    addition_schedule_task_1(
        std::vector<size_t> &nzorb, const symmetry<N, element_type> &syma,
        const symmetry<N, element_type> &symb,
        std::map<size_t, book_node> &booka, libutil::spinlock &lock) :

        m_syma(syma), m_symb(symb), m_booka(booka), m_lock(lock) {

        std::swap(nzorb, m_nzorb);
    }

    virtual ~addition_schedule_task_1() { }

    virtual void perform() {

        std::map<size_t, book_node> booka;

        for(size_t i = 0; i < m_nzorb.size(); i++) {
            size_t acia = m_nzorb[i];
            orbit<N, element_type> oa(m_syma, acia);
            subgroup_orbits<N, element_type> soa(m_syma, m_symb, acia);
            for(typename subgroup_orbits<N, element_type>::iterator i =
                soa.begin(); i != soa.end(); ++i) {

                size_t acic = soa.get_abs_index(i);
                book_node n;
                n.cidx = acia;
                n.tr = oa.get_transf(acic);
                n.visited = false;
                booka[acic] = n;
            }
        }

        {
            libutil::auto_lock<libutil::spinlock> lock(m_lock);
            m_booka.insert(booka.begin(), booka.end());
        }
    }

};


template<size_t N, typename Traits, typename Iterator>
class addition_schedule_task_iterator_1 : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename addition_schedule<N, Traits>::book_node book_node;

private:
    Iterator m_ibegin;
    Iterator m_iend;
    const symmetry<N, element_type> &m_syma;
    const symmetry<N, element_type> &m_symb;
    std::map<size_t, book_node> &m_booka;
    libutil::spinlock m_lock;

public:
    addition_schedule_task_iterator_1(
        const Iterator &ibegin, const Iterator &iend,
        const symmetry<N, element_type> &syma,
        const symmetry<N, element_type> &symb,
        std::map<size_t, book_node> &booka) :

        m_ibegin(ibegin), m_iend(iend), m_syma(syma), m_symb(symb),
        m_booka(booka) {

    }

    virtual bool has_more() const {
        return m_ibegin != m_iend;
    }

    virtual libutil::task_i *get_next() {

        const size_t batch_size = 10;

        std::vector<size_t> nzorb;
        nzorb.reserve(batch_size);

        size_t i = 0;
        while(m_ibegin != m_iend && i < batch_size) {
            nzorb.push_back(*m_ibegin);
            ++m_ibegin;
            i++;
        }

        return new addition_schedule_task_1<N, Traits>(nzorb, m_syma, m_symb,
            m_booka, m_lock);
    }

};


template<size_t N, typename Traits>
class addition_schedule_task_2 : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename addition_schedule<N, Traits>::node node;
    typedef typename addition_schedule<N, Traits>::book_node book_node;
    typedef typename addition_schedule<N, Traits>::schedule_group
        schedule_group;
    typedef typename addition_schedule<N, Traits>::schedule_type schedule_type;

private:
    std::vector<size_t> m_batch;
    const symmetry<N, element_type> &m_syma;
    const symmetry<N, element_type> &m_symb;
    const symmetry<N, element_type> &m_symc;
    std::map<size_t, book_node> &m_booka;
    std::map<size_t, book_node> &m_bookb;
    schedule_type &m_sch;
    libutil::spinlock &m_lock;

public:
    addition_schedule_task_2(
        std::vector<size_t> &batch,
        const symmetry<N, element_type> &syma,
        const symmetry<N, element_type> &symb,
        const symmetry<N, element_type> &symc,
        std::map<size_t, book_node> &booka,
        std::map<size_t, book_node> &bookb,
        schedule_type &sch,
        libutil::spinlock &lock) :

        m_syma(syma), m_symb(symb), m_symc(symc), m_booka(booka),
        m_bookb(bookb), m_sch(sch), m_lock(lock) {

        std::swap(m_batch, batch);
    }

    virtual ~addition_schedule_task_2() { }

    virtual void perform() {

        for(size_t i = 0; i < m_batch.size(); i++) {

            size_t acic = m_batch[i];

            combined_orbits<N, element_type> co(m_syma, m_symb, m_symc, acic);
            std::auto_ptr<schedule_group> grp(new schedule_group);
            bool first = true, already_visited = false;
            for(typename combined_orbits<N, element_type>::iterator i =
                co.begin(); i != co.end(); ++i) {

                size_t ic = co.get_abs_index(i);
                typename std::map<size_t, book_node>::iterator ia =
                    m_booka.find(ic);
                typename std::map<size_t, book_node>::iterator ib =
                    m_bookb.find(ic);

                if(ia == m_booka.end() && ib == m_bookb.end()) continue;
                if(first) {
                    libutil::auto_lock<libutil::spinlock> lock(m_lock);
                    if(ia != m_booka.end()) {
                        if(ia->second.visited) {
                            already_visited = true;
                        } else {
                            ia->second.visited = true;
                        }
                    } else {
                        if(ib->second.visited) {
                            already_visited = true;
                        } else {
                            ib->second.visited = true;
                        }
                    }
                    first = false;
                    if(already_visited) break;
                }

                node n;
                n.cic = ic;
                if(ia == m_booka.end()) {
                    n.zeroa = true;
                } else {
                    n.zeroa = false;
                    n.cia = ia->second.cidx;
                    n.tra = ia->second.tr;
                }
                if(ib == m_bookb.end()) {
                    n.zerob = true;
                } else {
                    n.zerob = false;
                    n.cib = ib->second.cidx;
                    n.trb = ib->second.tr;
                }
                if(!n.zeroa || !n.zerob) grp->push_back(n);
            }

            if(!already_visited) {
                libutil::auto_lock<libutil::spinlock> lock(m_lock);
                m_sch.push_back(grp.release());
            }
        }
    }

};


template<size_t N, typename Traits>
class addition_schedule_task_iterator_2 : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename addition_schedule<N, Traits>::book_node book_node;
    typedef typename addition_schedule<N, Traits>::schedule_type schedule_type;

private:
    const symmetry<N, element_type> &m_syma;
    const symmetry<N, element_type> &m_symb;
    const symmetry<N, element_type> &m_symc;
    std::map<size_t, book_node> &m_booka;
    std::map<size_t, book_node> &m_bookb;
    typename std::map<size_t, book_node>::iterator m_ia;
    typename std::map<size_t, book_node>::iterator m_ib;
    schedule_type &m_sch;
    libutil::spinlock m_lock;

public:
    addition_schedule_task_iterator_2(
        const symmetry<N, element_type> &syma,
        const symmetry<N, element_type> &symb,
        const symmetry<N, element_type> &symc,
        std::map<size_t, book_node> &booka,
        std::map<size_t, book_node> &bookb,
        schedule_type &sch) :

        m_syma(syma), m_symb(symb), m_symc(symc), m_booka(booka),
        m_bookb(bookb), m_ia(m_booka.begin()), m_ib(m_bookb.begin()),
        m_sch(sch) {

    }

    virtual bool has_more() const {

        return m_ia != m_booka.end() || m_ib != m_bookb.end();
    }

    virtual libutil::task_i *get_next() {

        const size_t batch_size = 10;

        std::vector<size_t> batch;
        batch.reserve(batch_size);

        {
            libutil::auto_lock<libutil::spinlock> lock(m_lock);

            while(batch.size() < batch_size && m_ia != m_booka.end()) {
                if(!m_ia->second.visited) batch.push_back(m_ia->first);
                ++m_ia;
            }
            while(batch.size() < batch_size && m_ib != m_bookb.end()) {
                if(!m_ib->second.visited) batch.push_back(m_ib->first);
                ++m_ib;
            }
        }

        return new addition_schedule_task_2<N, Traits>(batch, m_syma, m_symb,
            m_symc, m_booka, m_bookb, m_sch, m_lock);
    }

};


template<size_t N, typename Traits>
class addition_schedule_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t) {
        delete t;
    }

};


} // unnamed namespace


template<size_t N, typename Traits>
void addition_schedule<N, Traits>::build(
    const assignment_schedule_type &asch,
    gen_block_tensor_rd_ctrl<N, bti_traits> &cb) {

    typedef typename assignment_schedule_type::iterator asgsch_iterator;

    addition_schedule<N, Traits>::start_timer();

    try {

        clear_schedule();

        dimensions<N> bidims(m_syma.get_bis().get_block_index_dims());

        std::vector<size_t> nzlstb;
        cb.req_nonzero_blocks(nzlstb);

        std::map<size_t, book_node> booka, bookb;

        {
            typedef typename assignment_schedule_type::iterator iterator_type;
            addition_schedule_task_iterator_1<N, Traits, iterator_type> ti(
                asch.begin(), asch.end(), m_syma, m_symc, booka);
            addition_schedule_task_observer<N, Traits> to;
            libutil::thread_pool::submit(ti, to);
        }

        {
            typedef typename std::vector<size_t>::const_iterator iterator_type;
            addition_schedule_task_iterator_1<N, Traits, iterator_type> ti(
                nzlstb.begin(), nzlstb.end(), m_symb, m_symc, bookb);
            addition_schedule_task_observer<N, Traits> to;
            libutil::thread_pool::submit(ti, to);
        }

        {
            addition_schedule_task_iterator_2<N, Traits> ti(
                m_syma, m_symb, m_symc, booka, bookb, m_sch);
            addition_schedule_task_observer<N, Traits> to;
            libutil::thread_pool::submit(ti, to);
        }

    } catch(...) {
        addition_schedule<N, Traits>::stop_timer();
        throw;
    }

    addition_schedule<N, Traits>::stop_timer();
}


template<size_t N, typename Traits>
void addition_schedule<N, Traits>::clear_schedule() throw() {

    for(typename schedule_type::iterator i = m_sch.begin();
        i != m_sch.end(); ++i) delete *i;
    m_sch.clear();
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITION_SCHEDULE_IMPL_H
