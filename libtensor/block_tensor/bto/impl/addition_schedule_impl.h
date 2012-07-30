#ifndef LIBTENSOR_ADDITION_SCHEDULE_IMPL_H
#define LIBTENSOR_ADDITION_SCHEDULE_IMPL_H

#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include "../addition_schedule.h"

namespace libtensor {


template<size_t N, typename Traits>
const char *addition_schedule<N, Traits>::k_clazz =
    "addition_schedule<N, Traits>";


template<size_t N, typename Traits>
addition_schedule<N, Traits>::addition_schedule(const symmetry_type &syma,
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
void addition_schedule<N, Traits>::build(const assignment_schedule_type &asch,
    block_tensor_ctrl_type &cb) {

    typedef typename assignment_schedule_type::iterator asgsch_iterator;

    addition_schedule<N, Traits>::start_timer();

    clear_schedule();

    dimensions<N> bidims(m_syma.get_bis().get_block_index_dims());

    size_t sz = bidims.get_size();
    std::vector<char> oa(sz, 0), ob(sz, 0), omb(sz, 0), omc(sz, 0);
    mark_orbits(m_symb, omb);
    mark_orbits(m_symc, omc);

    for(asgsch_iterator i = asch.begin(); i != asch.end(); ++i) {

        abs_index<N> acia(asch.get_abs_index(i), bidims);

        schedule_group *grp = 0;
        typename book_t::iterator igrp = m_posta.find(acia.get_abs_index());
        if(igrp != m_posta.end()) {
            grp = igrp->second;
            m_posta.erase(igrp);
        } else {
            grp = new schedule_group;
            m_sch.push_back(grp);
        }
        m_booka.insert(book_pair_t(acia.get_abs_index(), grp));

        tensor_transf<N, element_type> tra0;
        process_orbit_in_a(bidims, false, cb, acia, acia, tra0, oa, ob,
            omb, omc, *grp);
    }

    for(typename book_t::iterator i = m_posta.begin();
        i != m_posta.end(); ++i) {

        abs_index<N> acia(i->first, bidims);
        tensor_transf<N, element_type> tra0;
        process_orbit_in_a(bidims, true, cb, acia, acia, tra0, oa, ob,
            omb, omc, *i->second);
    }

    schedule_group *extra = new schedule_group;
    for(size_t i = 0; i < sz; i++) {

        if(omb[i] != 2 || oa[i] != 0) continue;

        abs_index<N> acib(i, bidims);
        if(cb.req_is_zero_block(acib.get_index())) continue;

        tensor_transf<N, element_type> trb0;
        process_orbit_in_b(bidims, true, acib, acib, trb0, oa, ob,
            omb, omc, *extra);
    }
    if(!extra->empty()) m_sch.push_back(extra);
    else delete extra;

    m_booka.clear();
    m_posta.clear();

    addition_schedule<N, Traits>::stop_timer();
}


template<size_t N, typename Traits>
void addition_schedule<N, Traits>::clear_schedule() throw() {

    m_booka.clear();
    m_posta.clear();
    for(typename schedule_type::iterator i = m_sch.begin();
        i != m_sch.end(); ++i) delete *i;
    m_sch.clear();
}


template<size_t N, typename Traits>
void addition_schedule<N, Traits>::mark_orbits(const symmetry_type &sym,
    std::vector<char> &o) {

    dimensions<N> bidims(sym.get_bis().get_block_index_dims());
    abs_index<N> aci(bidims);
    do {
        if(o[aci.get_abs_index()] == 0) {
            o[aci.get_abs_index()] = 2;
            if(!mark_orbit_inner(sym, aci, o)) {
                o[aci.get_abs_index()] = 4;
            }
        }
    } while(aci.inc());
}


template<size_t N, typename Traits>
bool addition_schedule<N, Traits>::mark_orbit_inner(const symmetry_type &sym,
    const abs_index<N> &ai, std::vector<char> &o) {

    typedef symmetry_element_set<N, element_type> seset_type;
    typedef typename symmetry_type::iterator symmetry_iterator;
    typedef typename seset_type::const_iterator seset_iterator;

    bool allowed = true;
    for(symmetry_iterator is = sym.begin(); is != sym.end(); ++is) {
        const seset_type &es = sym.get_subset(is);
        for(seset_iterator ie = es.begin(); ie != es.end(); ++ie) {
            const symmetry_element_i<N, element_type> &e = es.get_elem(ie);
            index<N> i1(ai.get_index());
            allowed = allowed && e.is_allowed(i1);
            e.apply(i1);
            abs_index<N> ai1(i1, ai.get_dims());
            if(o[ai1.get_abs_index()] == 0) {
                o[ai1.get_abs_index()] = allowed ? 1 : 3;
                if(!mark_orbit_inner(sym, ai1, o)) {
                    allowed = false;
                    o[ai1.get_abs_index()] = 3;
                }
            }
        }
    }
    return allowed;
}


template<size_t N, typename Traits>
size_t addition_schedule<N, Traits>::find_canonical(const dimensions<N> &bidims,
    const symmetry_type &sym, const abs_index<N> ai, tensor_transf_type &tr,
    const std::vector<char> &o) {

    //  Given the orbit array o previously formed by mark_orbits(), this
    //  function returns the canonical index of the orbit to which ai belongs.
    //  It also applies the transformation from the canonical block to the
    //  given block to a provided tensor transformation tr.

    if(o[ai.get_abs_index()] == 2) return ai.get_abs_index();

    std::vector<char> o2(o.size(), 0);

    tensor_transf<N, element_type> tr1; // From current to canonical
    size_t ii = find_canonical_inner(bidims, sym, ai, tr1, o, o2);
    tr1.invert(); // From canonical to current
    tr.transform(tr1);
    return ii;
}


template<size_t N, typename Traits>
size_t addition_schedule<N, Traits>::find_canonical_inner(
    const dimensions<N> &bidims, const symmetry_type &sym,
    const abs_index<N> &ai, tensor_transf_type &tr, const std::vector<char> &o,
    std::vector<char> &o2) {

    //  Applies all possible combinations of symmetry elements until it
    //  explores the entire orbit. Array o2 is used to mark visited indexes.
    //  Returns the smallest index in the orbit (canonical index) and
    //  the corresponding transformation tr.

    typedef symmetry_element_set<N, element_type> seset_type;
    typedef typename symmetry_type::iterator symmetry_iterator;
    typedef typename seset_type::const_iterator seset_iterator;

    o2[ai.get_abs_index()] = 1;
    size_t smallest = ai.get_abs_index();

    for(symmetry_iterator is = sym.begin(); is != sym.end(); ++is) {
        const seset_type &es = sym.get_subset(is);
        for(seset_iterator ie = es.begin(); ie != es.end(); ++ie) {
            const symmetry_element_i<N, element_type> &e = es.get_elem(ie);
            index<N> i1(ai.get_index());
            tensor_transf_type tr1;
            e.apply(i1, tr1);
            abs_index<N> ai1(i1, bidims);
            if(o[ai1.get_abs_index()] == 2 || o[ai1.get_abs_index()] == 4) {
                tr.transform(tr1);
                return ai1.get_abs_index();
            }
            if(o2[ai1.get_abs_index()] == 0) {
                size_t ii = find_canonical_inner(bidims, sym, ai1, tr1, o, o2);
                if(o[ii] == 2 || o[ii] == 4) {
                    tr.transform(tr1);
                    return ii;
                }
                if(ii < smallest) smallest = ii;
            }
        }
    }
    return smallest;
}


template<size_t N, typename Traits>
void addition_schedule<N, Traits>::process_orbit_in_a(
    const dimensions<N> &bidims, bool zeroa, block_tensor_ctrl_type &cb,
    const abs_index<N> &acia, const abs_index<N> &aia,
    const tensor_transf_type &tra, std::vector<char> &oa, std::vector<char> &ob,
    const std::vector<char> &omb, const std::vector<char> &omc,
    schedule_group &grp) {

    if(oa[aia.get_abs_index()]) return;

    oa[aia.get_abs_index()] =
        (acia.get_abs_index() == aia.get_abs_index()) ? 2 : 1;

    //  Index in B and C that corresponds to the index in A
    index<N> ib(aia.get_index());
    abs_index<N> aib(ib, bidims);

    //  Process only allowed canonical blocks in C
    if(omc[aib.get_abs_index()] == 2) {

        bool cana = oa[aia.get_abs_index()] == 2;
        bool canb = omb[aib.get_abs_index()] == 2 ||
            omb[aib.get_abs_index()] == 4;

        tensor_transf_type trb;
        abs_index<N> acib(canb ? aib.get_abs_index() :
            find_canonical(bidims, m_symb, aib, trb, omb), bidims);
        bool allowedb = omb[acib.get_abs_index()] == 2;
        bool zerob = true;
        if(allowedb) zerob = cb.req_is_zero_block(acib.get_index());

        if(zeroa) {
            if(!zerob) {
                grp.push_back(node(acib.get_abs_index(),
                    aib.get_abs_index(), trb));
            }
        } else {
            grp.push_back(node(acia.get_abs_index(), acib.get_abs_index(),
                aib.get_abs_index(), tra, trb));
        }

        if(allowedb) {
            iterate_sym_elements_in_b(bidims, zeroa, acib, aib, trb, oa, ob,
                omb, omc, grp);
        }
    }

    //  Continue exploring the orbit recursively
    iterate_sym_elements_in_a(bidims, zeroa, cb, acia, aia, tra, oa, ob,
        omb, omc, grp);
}


template<size_t N, typename Traits>
void addition_schedule<N, Traits>::iterate_sym_elements_in_a(
    const dimensions<N> &bidims, bool zeroa, block_tensor_ctrl_type &cb,
    const abs_index<N> &acia, const abs_index<N> &aia,
    const tensor_transf_type &tra, std::vector<char> &oa, std::vector<char> &ob,
    const std::vector<char> &omb, const std::vector<char> &omc,
    schedule_group &grp) {

    typedef symmetry_element_set<N, element_type> seset_type;
    typedef typename symmetry_type::iterator symmetry_iterator;
    typedef typename seset_type::const_iterator seset_iterator;

    for(symmetry_iterator is = m_syma.begin(); is != m_syma.end(); ++is) {
        const seset_type &es = m_syma.get_subset(is);
        for(seset_iterator ie = es.begin(); ie != es.end(); ++ie) {
            const symmetry_element_i<N, element_type> &e = es.get_elem(ie);
            index<N> ia1(aia.get_index());
            tensor_transf_type tra1(tra);
            e.apply(ia1, tra1);
            abs_index<N> aia1(ia1, bidims);
            if(oa[aia1.get_abs_index()] == 0) {
                process_orbit_in_a(bidims, zeroa, cb, acia, aia1, tra1,
                    oa, ob, omb, omc, grp);
            }
        }
    }
}


template<size_t N, typename Traits>
void addition_schedule<N, Traits>::process_orbit_in_b(
    const dimensions<N> &bidims, bool zeroa, const abs_index<N> &acib,
    const abs_index<N> &aib, const tensor_transf_type &trb,
    std::vector<char> &oa, std::vector<char> &ob, const std::vector<char> &omb,
    const std::vector<char> &omc, schedule_group &grp) {

    if(ob[aib.get_abs_index()]) return;

    ob[aib.get_abs_index()] =
        (acib.get_abs_index() == aib.get_abs_index()) ? 2 : 1;

    //  Index in A and C that corresponds to the index in B
    index<N> ia(aib.get_index());
    abs_index<N> aia(ia, bidims);

    //  Process only allowed canonical blocks in C
    if(omc[aia.get_abs_index()] == 2) {

        bool cana = oa[aia.get_abs_index()] == 2;
        bool canb = omb[aib.get_abs_index()] == 2;

        tensor_transf_type tra;
        abs_index<N> acia(cana ? aia.get_abs_index() :
            find_canonical(bidims, m_syma, aia, tra, oa), bidims);

        typename book_t::iterator igrp = m_booka.find(acia.get_abs_index());
        if(igrp == m_booka.end()) {
            if(zeroa) {
                if(!canb) {
                    grp.push_back(node(acib.get_abs_index(),
                        aib.get_abs_index(), trb));
                }
            } else {
                m_posta.insert(book_pair_t(acia.get_abs_index(), &grp));
            }
        }
    }

    //  Continue exploring the orbit recursively
    iterate_sym_elements_in_b(bidims, zeroa, acib, aib, trb, oa, ob,
        omb, omc, grp);
}


template<size_t N, typename Traits>
void addition_schedule<N, Traits>::iterate_sym_elements_in_b(
    const dimensions<N> &bidims, bool zeroa, const abs_index<N> &acib,
    const abs_index<N> &aib, const tensor_transf_type &trb,
    std::vector<char> &oa, std::vector<char> &ob, const std::vector<char> &omb,
    const std::vector<char> &omc, schedule_group &grp) {

    typedef symmetry_element_set<N, element_type> seset_type;
    typedef typename symmetry_type::iterator symmetry_iterator;
    typedef typename seset_type::const_iterator seset_iterator;

    for(symmetry_iterator is = m_symb.begin(); is != m_symb.end(); ++is) {
        const seset_type &es = m_symb.get_subset(is);
        for(seset_iterator ie = es.begin(); ie != es.end(); ++ie) {
            const symmetry_element_i<N, element_type> &e = es.get_elem(ie);
            index<N> ib1(aib.get_index());
            tensor_transf_type trb1(trb);
            e.apply(ib1, trb1);
            abs_index<N> aib1(ib1, bidims);
            if(ob[aib1.get_abs_index()] == 0) {
                process_orbit_in_b(bidims, zeroa, acib, aib1, trb1, oa, ob,
                    omb, omc, grp);
            }
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITION_SCHEDULE_IMPL_H
