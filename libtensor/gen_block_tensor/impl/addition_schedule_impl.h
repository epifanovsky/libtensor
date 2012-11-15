#ifndef LIBTENSOR_ADDITION_SCHEDULE_IMPL_H
#define LIBTENSOR_ADDITION_SCHEDULE_IMPL_H

#include <cstring>
#include <algorithm>
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


template<size_t N, typename Traits>
void addition_schedule<N, Traits>::build(
    const assignment_schedule_type &asch,
    gen_block_tensor_rd_ctrl<N, bti_traits> &cb) {

    typedef typename assignment_schedule_type::iterator asgsch_iterator;

    addition_schedule<N, Traits>::start_timer();

    clear_schedule();

    dimensions<N> bidims(m_syma.get_bis().get_block_index_dims());

    std::vector<size_t> nzlstb;
    cb.req_nonzero_blocks(nzlstb);

    std::map<size_t, book_node> booka, bookb;

    for(asgsch_iterator ia = asch.begin(); ia != asch.end(); ++ia) {

        size_t acia = asch.get_abs_index(ia);
        orbit<N, element_type> oa(m_syma, acia);
        subgroup_orbits<N, element_type> soa(m_syma, m_symc, acia);

        for(typename subgroup_orbits<N, element_type>::iterator i = soa.begin();
            i != soa.end(); ++i) {

            size_t acic = soa.get_abs_index(i);
            book_node n;
            n.cidx = acia;
            n.tr = oa.get_transf(acic);
            booka[acic] = n;
        }
    }

    for(typename std::vector<size_t>::const_iterator ib = nzlstb.begin();
        ib != nzlstb.end(); ++ib) {

        size_t acib = *ib;
        orbit<N, element_type> ob(m_symb, acib);
        subgroup_orbits<N, element_type> sob(m_symb, m_symc, acib);

        for(typename subgroup_orbits<N, element_type>::iterator i = sob.begin();
            i != sob.end(); ++i) {

            size_t acic = sob.get_abs_index(i);
            book_node n;
            n.cidx = acib;
            n.tr = ob.get_transf(acic);
            bookb[acic] = n;
        }
    }

    while(!booka.empty() || !bookb.empty()) {
        size_t acic;
        if(!booka.empty()) acic = booka.begin()->first;
        else acic = bookb.begin()->first;
        combined_orbits<N, element_type> co(m_syma, m_symb, m_symc, acic);
        schedule_group *grp = new schedule_group;
        for(typename combined_orbits<N, element_type>::iterator i = co.begin();
            i != co.end(); ++i) {
            size_t ic = co.get_abs_index(i);
            typename std::map<size_t, book_node>::iterator ia = booka.find(ic);
            typename std::map<size_t, book_node>::iterator ib = bookb.find(ic);
            node n;
            n.cic = ic;
            if(ia == booka.end()) {
                n.zeroa = true;
            } else {
                n.zeroa = false;
                n.cia = ia->second.cidx;
                n.tra = ia->second.tr;
                booka.erase(ia);
            }
            if(ib == bookb.end()) {
                n.zerob = true;
            } else {
                n.zerob = false;
                n.cib = ib->second.cidx;
                n.trb = ib->second.tr;
                bookb.erase(ib);
            }
            if(!n.zeroa || !n.zerob) grp->push_back(n);
        }
        m_sch.push_back(grp);
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
