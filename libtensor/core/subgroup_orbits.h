#ifndef LIBTENSOR_SUBGROUP_ORBITS_H
#define LIBTENSOR_SUBGROUP_ORBITS_H

#include <cstdlib> // for size_t
#include <algorithm> // for std::binary_search
#include <vector>
#include "abs_index.h"
#include "dimensions.h"
#include "index.h"
#include "magic_dimensions.h"
#include "noncopyable.h"
#include "symmetry.h"

namespace libtensor {


/** \brief Given an orbit in a group, builds a list of orbits in a subgroup
    \tparam N Tensor order.
    \tparam T Tensor element type.

    Given groups G1, G2: G2 is a subgroup of G1, and index I in C1, where
    C1 is a set of all canonical indexes in G1 and
    C2 is a set of all canonical indexes in G2,
    this algorithm finds a set J: {Ji | Ji in C2, Ji in orbit(I, G1)}.

    The output set J is sorted by absolute value of Ji and is accessible using
    an iterator.

    \sa orbit, orbit_list, combined_orbits

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class subgroup_orbits : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename std::vector<size_t>::const_iterator iterator;

private:
    dimensions<N> m_dims; //!< Index dimensions
    magic_dimensions<N> m_mdims; //!< Magic dimensions
    std::vector<size_t> m_orb; //!< Sorted vector of canonical indexes Ji

public:
    /** \brief Constructs the list of orbits
        \param sym1 Symmetry group G1.
        \param sym2 Symmetry group G2 (must be subgroup of G1).
        \param aidx Absolute value of index I.
     **/
    subgroup_orbits(
        const symmetry<N, T> &sym1,
        const symmetry<N, T> &sym2,
        size_t aidx);

    /** \brief Returns the number of orbits on the list
     **/
    size_t get_size() const {
        return m_orb.size();
    }

    /** \brief Returns true is the given index is a canonical one and contained
            on the list
        \param idx Index.
     **/
    bool contains(const index<N> &idx) const {
        return contains(abs_index<N>::get_abs_index(idx, m_dims));
    }

    /** \brief Returns true is the given index is a canonical one and contained
            on the list
        \param aidx Absolute value of an index.
     **/
    bool contains(size_t aidx) const {
        return std::binary_search(m_orb.begin(), m_orb.end(), aidx);
    }

    /** \brief Returns an STL-like iterator to the beginning of the orbit list
     **/
    iterator begin() const {
        return m_orb.begin();
    }

    /** \brief Returns an STL-like iterator to the end of the orbit list
     **/
    iterator end() const {
        return m_orb.end();
    }

    /** \brief Returns the absolute value of a canonical index pointed to by
            the iterator
     **/
    size_t get_abs_index(const iterator &i) const {
        return *i;
    }

    /** \brief Returns the canonical index pointed to by the iterator
        \param[out] idx Canonical index.
     **/
    void get_index(const iterator &i, index<N> &idx) const {
        abs_index<N>::get_index(*i, m_dims, idx);
    }

private:
    void build_orbit(const symmetry<N, T> &sym, size_t aidx,
        std::vector<size_t> &orb);

};


} // namespace libtensor

#endif // LIBTENSOR_SUBGROUP_ORBITS_H
