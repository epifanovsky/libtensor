#ifndef LIBTENSOR_ADDITION_SCHEDULE_H
#define LIBTENSOR_ADDITION_SCHEDULE_H

#include <list>
#include <vector>
#include <libtensor/timings.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_index_space.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/core/symmetry.h>
#include "gen_block_tensor_i.h"
#include "assignment_schedule.h"

namespace libtensor {


/** \brief Creates a schedule for the block-wise addition of two block tensors
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits

    The symmetry group of the sum C of two tensors A and B is the largest common
    subgroup of the arguments. Therefore, each orbit in the arguments maps to
    one or more orbits in the result, but each orbit in C maps to only one orbit
    in A and one orbit in B.

    Based on these orbit mappings, this symmetry algorithm produces a table
    (schedule) that can be used to combine the blocks of tensors A and B into
    their sum C.

    The smallest schedule items (nodes) contain prescriptions on how to build
    a canonical block in C from two canonical blocks from A and B. The nodes are
    combined into disjoint groups. The set (array) of all the groups forms
    the addition schedule.

    The algorithm runs in a linear with the number of blocks time. This
    implementation is thread-safe is input symmetries of A and B remain
    unchanged in the course of running the algorithm.

    \ingroup libtensor_block_tensor_bto
 **/
template<size_t N, class Traits>
class addition_schedule :
    public timings< addition_schedule<N, Traits> >, public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

    typedef symmetry<N, element_type> symmetry_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;
    typedef assignment_schedule<N, element_type> assignment_schedule_type;

public:
    struct node {
        bool zeroa;
        size_t cia, cib, cic;
        tensor_transf_type tra, trb;

        node(size_t cia_, size_t cib_, size_t cic_,
            const tensor_transf_type &tra_, const tensor_transf_type &trb_) :
            zeroa(false), cia(cia_), cib(cib_), cic(cic_), tra(tra_), trb(trb_)
            { }

        node(size_t cib_, size_t cic_, const tensor_transf_type &trb_) :
            zeroa(true), cia(0), cib(cib_), cic(cic_), trb(trb_)
            { }

    };

    typedef std::list<node> schedule_group;
    typedef std::vector<schedule_group*> schedule_type; //!< Schedule type
    typedef typename schedule_type::const_iterator iterator;

    typedef std::pair<size_t, schedule_group*> book_pair_t;
    typedef std::map<size_t, schedule_group*> book_t;

private:
    const symmetry_type &m_syma; //!< Symmetry of A
    const symmetry_type &m_symb; //!< Symmetry of B
    symmetry_type m_symc; //!< Largest common subgroup of A and B
    schedule_type m_sch; //!< Additive schedule
    book_t m_booka, m_posta;

public:
    /** \brief Initializes the algorithm
     **/
    addition_schedule(
        const symmetry_type &syma,
        const symmetry_type &symb);

    /** \brief Destructor
     **/
    ~addition_schedule();

    /** \brief Runs the algorithm
     **/
    void build(
        const assignment_schedule_type &asch,
        gen_block_tensor_rd_ctrl<N, bti_traits> &cb);

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
    /** \brief Removes all elements from the schedule
     **/
    void clear_schedule() throw();

    /** \brief Fills output array with tags for each block index:
            1 = allowed non-canonical, 2 = allowed canonical,
            3 = forbidden non-canonical, 4 = forbidden canonical
     **/
    void mark_orbits(
        const symmetry_type &sym,
        std::vector<char> &o);

    /** \brief Returns the canonical index and a transformation to a given index
     **/
    size_t find_canonical(
        const dimensions<N> &bidims,
        const symmetry_type &sym,
        const abs_index<N> ai,
        tensor_transf_type &tr,
        const std::vector<char> &o);

    /** \brief Recursive part of find_canonical()
     **/
    size_t find_canonical_inner(
        const dimensions<N> &bidims,
        const symmetry_type &sym,
        const abs_index<N> &ai,
        tensor_transf_type &tr,
        const std::vector<char> &o,
        std::vector<char> &o2);

    void process_orbit_in_a(
        const dimensions<N> &bidims,
        bool zeroa,
        gen_block_tensor_rd_ctrl<N, bti_traits> &cb,
        const abs_index<N> &acia,
        const abs_index<N> &aia,
        const tensor_transf_type &tra,
        std::vector<char> &oa,
        std::vector<char> &ob,
        const std::vector<char> &omb,
        const std::vector<char> &omc,
        schedule_group &grp);

    void iterate_sym_elements_in_a(
        const dimensions<N> &bidims,
        bool zeroa,
        gen_block_tensor_rd_ctrl<N, bti_traits> &cb,
        const abs_index<N> &acia,
        const abs_index<N> &aia,
        const tensor_transf_type &tra,
        std::vector<char> &oa,
        std::vector<char> &ob,
        const std::vector<char> &omb,
        const std::vector<char> &omc,
        schedule_group &grp);

    void process_orbit_in_b(
        const dimensions<N> &bidims,
        bool zeroa,
        const abs_index<N> &acib,
        const abs_index<N> &aib,
        const tensor_transf_type &trb,
        std::vector<char> &oa,
        std::vector<char> &ob,
        const std::vector<char> &omb,
        const std::vector<char> &omc,
        schedule_group &grp);

    void iterate_sym_elements_in_b(
        const dimensions<N> &bidims,
        bool zeroa,
        const abs_index<N> &acib,
        const abs_index<N> &aib,
        const tensor_transf<N, element_type> &trb,
        std::vector<char> &oa, std::vector<char> &ob,
        const std::vector<char> &omb,
        const std::vector<char> &omc,
        schedule_group &grp);

};


} // namespace libtensor

#endif // LIBTENSOR_ADDITION_SCHEDULE_H
