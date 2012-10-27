#ifndef LIBTENSOR_ORBIT_LIST_H
#define LIBTENSOR_ORBIT_LIST_H

#include <cstring> // for size_t
#include <map>
#include <vector>
#include <libtensor/timings.h>
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
    typedef typename std::map< size_t, index<N> >::const_iterator iterator;

private:
    dimensions<N> m_dims; //!< Index dimensions
    std::map< size_t, index<N> > m_orb; //!< List of orbits

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
    bool contains(const index<N> &idx) const;

    /** \brief Returns true is the given index is a canonical one and contained
            on the list
        \param aidx Absolute value of an index.
     **/
    bool contains(size_t aidx) const;

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
        return i->first;
    }

    /** \brief Returns the canonical index pointed to by the iterator
     **/
    const index<N> &get_index(const iterator &i) const {
        return i->second;
    }

private:
    bool mark_orbit(const symmetry<N, T> &sym, const index<N> &idx,
        std::vector<char> &chk);

};


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_LIST_H
