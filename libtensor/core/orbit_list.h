#ifndef LIBTENSOR_ORBIT_LIST_H
#define LIBTENSOR_ORBIT_LIST_H

#include <cstdlib> // for size_t
#include <algorithm> // for std::binary_search
#include <vector>
#include <libtensor/timings.h>
#include "abs_index.h"
#include "dimensions.h"
#include "index.h"
#include "noncopyable.h"
#include "symmetry.h"

namespace libtensor {


/** \brief Builds list of orbits in a given symmetry
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This algorithm takes a symmetry object and constructs a list of canonical
    indexes in that symmetry. The list of orbits represented by their canonical
    indexes can be then iterated over using STL-like iterators.

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class orbit_list : public noncopyable, public timings< orbit_list<N, T> > {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename std::vector<size_t>::const_iterator iterator;

private:
    dimensions<N> m_dims; //!< Index dimensions
    magic_dimensions<N> m_mdims; //!< Magic dimensions
    std::vector<size_t> m_orb; //!< Sorted vector of canonical indexes

public:
    /** \brief Constructs the list of orbits
        \param sym Symmetry group
     **/
    orbit_list(const symmetry<N, T> &sym);

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
    bool mark_orbit(const symmetry<N, T> &sym, size_t aidx0,
        std::vector<char> &chk);

};


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_LIST_H
