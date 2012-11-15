#ifndef LIBTENSOR_COMBINED_ORBITS_H
#define LIBTENSOR_COMBINED_ORBITS_H

#include <cstring> // for size_t
#include <algorithm> // for std::binary_search
#include <vector>
#include "abs_index.h"
#include "dimensions.h"
#include "index.h"
#include "noncopyable.h"
#include "symmetry.h"

namespace libtensor {


/** \brief Given an orbit in two groups, builds a list of orbits in their
        subgroup
    \tparam N Tensor order.
    \tparam T Tensor element type.

    Given groups G1, G2, G3: G3 is a subgroup of G1 and G2, and index I,
    this algorithm finds a complete set of indexes Ji:
    {Ji | Ji in C3, Ji in orbit(I, G1), Ji in orbit(I, G2)},
    where C3 is the set of all canonical indexes in G3.

    The output set J is sorted by absolute value of Ji and is accessible using
    an iterator.

    The result of this algorithm is the same as subgroup_orbits with G0 and G3,
    where G0 is the union of G1 and G2.

    \sa orbit, orbit_list, subgroup_orbits

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class combined_orbits : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename std::vector<size_t>::const_iterator iterator;

private:
    dimensions<N> m_dims; //!< Index dimensions
    std::vector<size_t> m_orb; //!< Sorted vector of canonical indexes Ji

public:
    /** \brief Constructs the list of orbits
        \param sym1 Symmetry group G1.
        \param sym2 Symmetry group G2.
        \param sym3 Symmetry group G3 (must be subgroup of both G1 and G2).
        \param aidx Absolute value of index I.
     **/
    combined_orbits(
        const symmetry<N, T> &sym1,
        const symmetry<N, T> &sym2,
        const symmetry<N, T> &sym3,
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

#endif // LIBTENSOR_COMBINED_ORBITS_H
