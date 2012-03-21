#ifndef LIBTENSOR_ADDITION_SCHEDULE_H
#define LIBTENSOR_ADDITION_SCHEDULE_H

#include <list>
#include <vector>
#include "transf_double.h"
#include "../core/abs_index.h"
#include "../core/block_index_space.h"
#include "../core/block_index_space_product_builder.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/symmetry.h"
#include "../symmetry/so_dirsum.h"
#include "../symmetry/so_merge.h"
#include "assignment_schedule.h"

namespace libtensor {


/**	\brief Creates a schedule for the blockwise addition of two block
            tensors
        \tparam N Tensor order.

        Given the block index space and symmetries of two block tensors,
        this routine creates a schedule for the blockwise addition. The schedule
        is sorted such that the blocks of the first operand are accessed only
        once.

        Block %tensor operations shall use this routine to implement their
        additive interface.

        \ingroup libtensor_btod
 **/
template<size_t N, typename T>
class addition_schedule {
public:
    struct schedule_node {
        bool zeroa;
        size_t cia, cib, cic;
        transf<N, T> tra, trb;
        schedule_node(size_t cia_, size_t cib_, size_t cic_,
                const transf<N, T> &tra_, const transf<N, T> &trb_) :
                    zeroa(false), cia(cia_), cib(cib_), cic(cic_),
                    tra(tra_), trb(trb_) { }
        schedule_node(size_t cib_, size_t cic_,
                const transf<N, T> &trb_) : zeroa(true), cia(0),
                cib(cib_), cic(cic_), trb(trb_) { }
    };

    struct schedule_group {
        std::list<schedule_node> lst;
    };

    typedef std::list<schedule_group*> schedule_t; //!< Schedule type
    typedef typename schedule_t::const_iterator iterator;

    typedef std::pair<size_t, schedule_group*> book_pair_t;
    typedef std::map<size_t, schedule_group*> book_t;

private:
    const symmetry<N, T> &m_syma; //!< Symmetry of A
    const symmetry<N, T> &m_symb; //!< Symmetry of B
    symmetry<N, T> m_symc; //!< Largest common subgroup of A and B
    schedule_t m_sch; //!< Additive schedule
    book_t m_booka, m_posta;

public:
    /**	\brief Initializes the algorithm
     **/
    addition_schedule(const symmetry<N, T> &syma,
            const symmetry<N, T> &symb);

    /**	\brief Destructor
     **/
    ~addition_schedule();

    /**	\brief Runs the algorithm
     **/
    void build(const assignment_schedule<N, T> &asch,
            block_tensor_ctrl<N, T> &ctrlb);

    iterator begin() const {
        return m_sch.begin();
    }

    iterator end() const {
        return m_sch.end();
    }

    const schedule_group &get_node(const iterator &i) const {
        return **i;
    }

private:
    /**	\brief Removes all elements from the schedule
     **/
    void clean_schedule() throw();

    /**	\brief Puts 2 in the positions corresponding to canonical
                indexes and 1 for non-canonical indexes
     **/
    void mark_orbits(const symmetry<N, T> &sym, std::vector<char> &o);

    /**	\brief Recursive part of mark_orbits()
     **/
    bool mark_orbit(const symmetry<N, T> &sym, const abs_index<N> &aci,
            std::vector<char> &o);

    /**	\brief Returns the canonical %index and a transformation
                to a given %index
     **/
    size_t find_canonical(const dimensions<N> &bidims,
            const symmetry<N, T> &sym, const abs_index<N> ai,
            transf<N, T> &tr, const std::vector<char> &o);

    size_t find_canonical_iterate(const dimensions<N> &bidims,
            const symmetry<N, T> &sym, const abs_index<N> &ai,
            transf<N, T> &tr, const std::vector<char> &o,
            std::vector<char> &o2);

    void process_orbit_in_a(const dimensions<N> &bidims, bool zeroa,
            block_tensor_ctrl<N, double> &ctrlb,
            const abs_index<N> &acia, const abs_index<N> &aia,
            const transf<N, T> &tra,
            std::vector<char> &oa, std::vector<char> &ob,
            const std::vector<char> &omb, const std::vector<char> &omc,
            schedule_group &grp);

    void iterate_sym_elements_in_a(const dimensions<N> &bidims, bool zeroa,
            block_tensor_ctrl<N, double> &ctrlb,
            const abs_index<N> &acia, const abs_index<N> &aia,
            const transf<N, T> &tra,
            std::vector<char> &oa, std::vector<char> &ob,
            const std::vector<char> &omb, const std::vector<char> &omc,
            schedule_group &grp);

    void process_orbit_in_b(const dimensions<N> &bidims, bool zeroa,
            const abs_index<N> &acib, const abs_index<N> &aib,
            const transf<N, T> &trb,
            std::vector<char> &oa, std::vector<char> &ob,
            const std::vector<char> &omb, const std::vector<char> &omc,
            schedule_group &grp);

    void iterate_sym_elements_in_b(const dimensions<N> &bidims, bool zeroa,
            const abs_index<N> &acib, const abs_index<N> &aib,
            const transf<N, T> &trb,
            std::vector<char> &oa, std::vector<char> &ob,
            const std::vector<char> &omb, const std::vector<char> &omc,
            schedule_group &grp);

private:
    addition_schedule(const addition_schedule<N, T>&);
    const addition_schedule<N, T> &operator=(
            const addition_schedule<N, T>&);
};


template<size_t N, typename T>
addition_schedule<N, T>::addition_schedule(const symmetry<N, T> &syma,
        const symmetry<N, T> &symb) :

        m_syma(syma), m_symb(symb), m_symc(m_symb.get_bis()) {

//    permutation<N> perm0;
//    so_add<N, T>(m_syma, perm0, m_symb, perm0).perform(m_symc);
    permutation<N + N> perm0;
    block_index_space_product_builder<N, N> bbx(m_syma.get_bis(),
            m_symb.get_bis(), perm0);
    symmetry<N + N, T> symx(bbx.get_bis());
    so_dirsum<N, N, T>(m_syma, m_symb, perm0).perform(symx);
    so_merge<N + N, N + N, N, T> merge(symx);
    for (size_t i = 0; i < N; i++) {
        mask<N + N> m;
        m[i] = m[i + N] = true;
        merge.add_mask(m);
    }
    merge.perform(m_symc);
}


template<size_t N, typename T>
addition_schedule<N, T>::~addition_schedule() {

    clean_schedule();
}


template<size_t N, typename T>
void addition_schedule<N, T>::build(const assignment_schedule<N, T> &asch,
        block_tensor_ctrl<N, T> &ctrlb) {

    clean_schedule();
    dimensions<N> bidims(m_syma.get_bis().get_block_index_dims());

    size_t sz = bidims.get_size();
    std::vector<char> oa(sz, 0), ob(sz, 0), omb(sz, 0), omc(sz, 0);
    mark_orbits(m_symb, omb);
    mark_orbits(m_symc, omc);

    //~ std::cout << "process schedule in a" << std::endl;
    for(typename assignment_schedule<N, T>::iterator i = asch.begin();
            i != asch.end(); i++) {

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

        transf<N, T> tra0;
        process_orbit_in_a(bidims, false, ctrlb, acia, acia, tra0, oa,
                ob, omb, omc, *grp);
    }
    //~ std::cout << "process postponed a" << std::endl;
    for(typename book_t::iterator i = m_posta.begin(); i != m_posta.end();
            i++) {

        abs_index<N> acia(i->first, bidims);
        transf<N, T> tra0;
        process_orbit_in_a(bidims, true, ctrlb, acia, acia, tra0, oa,
                ob, omb, omc, *i->second);
    }
    //~ std::cout << "process non-zero blocks b" << std::endl;
    schedule_group *extra = new schedule_group;
    for(size_t i = 0; i < sz; i++) {

        if(omb[i] != 2 || oa[i] != 0) continue;

        abs_index<N> acib(i, bidims);
        if(ctrlb.req_is_zero_block(acib.get_index())) continue;

        transf<N, T> trb0;
        process_orbit_in_b(bidims, true, acib, acib, trb0, oa, ob,
                omb, omc, *extra);
    }
    if(!extra->lst.empty()) m_sch.push_back(extra);
    else delete extra;

    m_booka.clear();
    m_posta.clear();
    //~ std::cout << "addition schedule done" << std::endl;
}

template<size_t N, typename T>
void addition_schedule<N, T>::clean_schedule() throw() {

    m_booka.clear();
    m_posta.clear();
    for(typename schedule_t::iterator i = m_sch.begin();
            i != m_sch.end(); i++) delete *i;
    m_sch.clear();
}


template<size_t N, typename T>
void addition_schedule<N, T>::mark_orbits(const symmetry<N, T> &sym,
        std::vector<char> &o) {

    dimensions<N> bidims(sym.get_bis().get_block_index_dims());
    abs_index<N> aci(bidims);
    do {
        if(o[aci.get_abs_index()] == 0) {
            o[aci.get_abs_index()] = 2;
            if(!mark_orbit(sym, aci, o)) {
                o[aci.get_abs_index()] = 4;
            }
        }
    } while(aci.inc());
}


template<size_t N, typename T>
bool addition_schedule<N, T>::mark_orbit(const symmetry<N, T> &sym,
        const abs_index<N> &ai, std::vector<char> &o) {

    bool allowed = true;
    for(typename symmetry<N, T>::iterator is = sym.begin();
            is != sym.end(); is++) {

        const symmetry_element_set<N, T> &es = sym.get_subset(is);

        for(typename symmetry_element_set<N, T>::const_iterator ie =
                es.begin(); ie != es.end(); ie++) {

            const symmetry_element_i<N, T> &e = es.get_elem(ie);
            index<N> i1(ai.get_index());
            allowed = allowed && e.is_allowed(i1);
            e.apply(i1);
            abs_index<N> ai1(i1, ai.get_dims());
            if(o[ai1.get_abs_index()] == 0) {
                o[ai1.get_abs_index()] = allowed ? 1 : 3;
                if(!mark_orbit(sym, ai1, o)) {
                    allowed = false;
                    o[ai1.get_abs_index()] = 3;
                }
            }
        }
    }
    return allowed;
}


template<size_t N, typename T>
size_t addition_schedule<N, T>::find_canonical(const dimensions<N> &bidims,
        const symmetry<N, T> &sym, const abs_index<N> ai, transf<N, T> &tr,
        const std::vector<char> &o) {

    if(o[ai.get_abs_index()] == 2) return ai.get_abs_index();

    std::vector<char> o2(o.size(), 0);

    transf<N, T> tr1; // From current to canonical
    size_t ii = find_canonical_iterate(bidims, sym, ai, tr1, o, o2);
    tr1.invert(); // From canonical to current
    tr.transform(tr1);
    return ii;
}


template<size_t N, typename T>
size_t addition_schedule<N, T>::find_canonical_iterate(
        const dimensions<N> &bidims, const symmetry<N, T> &sym,
        const abs_index<N> &ai, transf<N, T> &tr,
        const std::vector<char> &o, std::vector<char> &o2) {

    o2[ai.get_abs_index()] = 1;
    size_t smallest = ai.get_abs_index();

    for(typename symmetry<N, T>::iterator is = sym.begin();
            is != sym.end(); is++) {

        const symmetry_element_set<N, T> &es = sym.get_subset(is);

        for(typename symmetry_element_set<N, T>::const_iterator ie =
                es.begin(); ie != es.end(); ie++) {

            const symmetry_element_i<N, T> &e = es.get_elem(ie);
            index<N> i1(ai.get_index());
            transf<N, T> tr1;
            e.apply(i1, tr1);
            abs_index<N> ai1(i1, bidims);
            if(o[ai1.get_abs_index()] == 2 || o[ai1.get_abs_index()] == 4) {
                tr.transform(tr1);
                return ai1.get_abs_index();
            }
            if(o2[ai1.get_abs_index()] == 0) {
                size_t ii = find_canonical_iterate(bidims,
                        sym, ai1, tr1, o, o2);
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


template<size_t N, typename T>
void addition_schedule<N, T>::process_orbit_in_a(const dimensions<N> &bidims,
        bool zeroa, block_tensor_ctrl<N, double> &ctrlb,
        const abs_index<N> &acia, const abs_index<N> &aia,
        const transf<N, T> &tra, std::vector<char> &oa, std::vector<char> &ob,
        const std::vector<char> &omb, const std::vector<char> &omc,
        schedule_group &grp) {

    if(oa[aia.get_abs_index()]) return;

    oa[aia.get_abs_index()] =
            acia.get_abs_index() == aia.get_abs_index() ? 2 : 1;

    //
    //	Index in B and C that corresponds to the index in A
    //
    index<N> ib(aia.get_index());
    abs_index<N> aib(ib, bidims);

    //
    //	Skip all unallowed and non-canonical blocks in C
    //
    if(omc[aib.get_abs_index()] == 2) {

        bool cana = oa[aia.get_abs_index()] == 2;
        bool canb = omb[aib.get_abs_index()] == 2 ||
                omb[aib.get_abs_index()] == 4;

        transf<N, T> trb;
        abs_index<N> acib(canb ? aib.get_abs_index() :
                find_canonical(bidims, m_symb, aib, trb, omb), bidims);
        bool allowedb = omb[acib.get_abs_index()] == 2;
        bool zerob = true;
        if(allowedb) zerob = ctrlb.req_is_zero_block(acib.get_index());

        //~ std::cout << "(a) " << &grp << " C" << aib.get_index() << " - A" << acia.get_index() << (zeroa ? "=0 " : "   ") << "B" << acib.get_index();

        if(zeroa) {
            if(!zerob) {
                grp.lst.push_back(schedule_node(acib.get_abs_index(),
                        aib.get_abs_index(), trb));
                //~ std::cout << " - inserted in " << &grp << std::endl;
            } else {
                //~ std::cout << " - skipped" << std::endl;
            }
        } else {
            grp.lst.push_back(schedule_node(acia.get_abs_index(),
                    acib.get_abs_index(), aib.get_abs_index(),
                    tra, trb));
            //~ std::cout << " - inserted in " << &grp << std::endl;
        }

        if(allowedb) {
            iterate_sym_elements_in_b(bidims, zeroa, acib, aib, trb,
                    oa, ob, omb, omc, grp);
        }
    }

    //
    //	Continue exploring the orbit recursively
    //
    iterate_sym_elements_in_a(bidims, zeroa, ctrlb, acia, aia, tra, oa, ob,
            omb, omc, grp);
}


template<size_t N, typename T>
void addition_schedule<N, T>::iterate_sym_elements_in_a(
        const dimensions<N> &bidims, bool zeroa,
        block_tensor_ctrl<N, double> &ctrlb, const abs_index<N> &acia,
        const abs_index<N> &aia, const transf<N, T> &tra,
        std::vector<char> &oa, std::vector<char> &ob,
        const std::vector<char> &omb, const std::vector<char> &omc,
        schedule_group &grp) {

    for(typename symmetry<N, T>::iterator is = m_syma.begin();
            is != m_syma.end(); is++) {

        const symmetry_element_set<N, T> &es = m_syma.get_subset(is);

        for(typename symmetry_element_set<N, T>::const_iterator ie =
                es.begin(); ie != es.end(); ie++) {

            const symmetry_element_i<N, T> &e = es.get_elem(ie);
            index<N> ia1(aia.get_index());
            transf<N, T> tra1(tra);
            e.apply(ia1, tra1);
            abs_index<N> aia1(ia1, bidims);
            if(oa[aia1.get_abs_index()] == 0) {
                process_orbit_in_a(bidims, zeroa, ctrlb, acia,
                        aia1, tra1, oa, ob, omb, omc, grp);
            }
        }
    }
}


template<size_t N, typename T>
void addition_schedule<N, T>::process_orbit_in_b(const dimensions<N> &bidims,
        bool zeroa, const abs_index<N> &acib, const abs_index<N> &aib,
        const transf<N, T> &trb, std::vector<char> &oa, std::vector<char> &ob,
        const std::vector<char> &omb, const std::vector<char> &omc,
        schedule_group &grp) {

    if(ob[aib.get_abs_index()]) return;

    ob[aib.get_abs_index()] =
            acib.get_abs_index() == aib.get_abs_index() ? 2 : 1;

    //
    //	Index in A and C that corresponds to the index in B
    //
    index<N> ia(aib.get_index());
    abs_index<N> aia(ia, bidims);

    //
    //	Skip all non-canonical blocks in C
    //
    if(omc[aia.get_abs_index()] == 2) {

        bool cana = oa[aia.get_abs_index()] == 2;
        bool canb = omb[aib.get_abs_index()] == 2;

        transf<N, T> tra;
        abs_index<N> acia(cana ? aia.get_abs_index() :
                find_canonical(bidims, m_syma, aia, tra, oa), bidims);

        //~ std::cout << "(b) " << &grp << " C" << aia.get_index() << " - A" << acia.get_index() << (zeroa ? "=0 " : "   ") << "B" << acib.get_index();

        typename book_t::iterator igrp = m_booka.find(acia.get_abs_index());
        if(igrp == m_booka.end()) {
            if(zeroa) {
                if(!canb) {
                    grp.lst.push_back(schedule_node(acib.get_abs_index(),
                            aib.get_abs_index(), trb));
                    //~ std::cout << " - inserted in " << &grp << std::endl;
                } else {
                    //~ std::cout << " - skipped" << std::endl;
                }
            } else {
                m_posta.insert(book_pair_t(acia.get_abs_index(), &grp));
                //~ std::cout << " - stub inserted (postpone)" << std::endl;
            }
        } else {
            //~ std::cout << " - found in " << igrp->second << std::endl;
        }
    }

    //
    //	Continue exploring the orbit recursively
    //
    iterate_sym_elements_in_b(bidims, zeroa, acib, aib, trb, oa, ob, omb,
            omc, grp);
}


template<size_t N, typename T>
void addition_schedule<N, T>::iterate_sym_elements_in_b(
        const dimensions<N> &bidims, bool zeroa, const abs_index<N> &acib,
        const abs_index<N> &aib, const transf<N, T> &trb,
        std::vector<char> &oa, std::vector<char> &ob,
        const std::vector<char> &omb, const std::vector<char> &omc,
        schedule_group &grp) {

    for(typename symmetry<N, T>::iterator is = m_symb.begin();
            is != m_symb.end(); is++) {

        const symmetry_element_set<N, T> &es = m_symb.get_subset(is);

        for(typename symmetry_element_set<N, T>::const_iterator ie =
                es.begin(); ie != es.end(); ie++) {

            const symmetry_element_i<N, T> &e = es.get_elem(ie);
            index<N> ib1(aib.get_index());
            transf<N, T> trb1(trb);
            e.apply(ib1, trb1);
            abs_index<N> aib1(ib1, bidims);
            if(ob[aib1.get_abs_index()] == 0) {
                process_orbit_in_b(bidims, zeroa, acib, aib1,
                        trb1, oa, ob, omb, omc, grp);
            }
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITION_SCHEDULE_H
