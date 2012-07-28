#ifndef LIBTENSOR_ORBIT_H
#define LIBTENSOR_ORBIT_H

#include <map>
#include <libtensor/timings.h>
#include "abs_index.h"
#include "tensor_transf.h"
#include "symmetry.h"

namespace libtensor {


/** \brief Computes a set of symmetry-equivalent blocks of a block tensor
        using one member of the set

    The action of the index symmetry group on the set of all block indexes in
    a block tensor generates an index set partition, each subset being an orbit.
    The smallest index in an orbit is its canonical index and the corresponding
    block is called the canonical block. All the blocks in a given orbit are
    connected to the canonical block via the elements of the symmetry group.
    As such, any block from the orbit can be obtained from the canonical block
    by applying a tensor transformation. In other words, all the blocks in
    one orbit are symmetry-equivalent, and knowing just the canonical block
    and the index symmetry group is sufficient to restore the entire orbit.

    Beginning from a starter block index, this symmetry algorithm computes
    the indexes of all the blocks that belong to the same orbit. Each index is
    accompanied by a tensor transformation from the canonical block to
    the target block.

    The orbit may be allowed or disallowed by the symmetry elements in
    the group. For the whole orbit to be allowed, all the indexes must be
    allowed, otherwise the orbit is not allowed and all its blocks are zero.

    The algorithm runs in a constant time, which depends on the size of
    the symmetry group. The algorithm is thread-safe as long as the symmetry
    is not altered in the course of constructing the orbit.

    \sa symmetry_i, orbit_list

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class orbit : public timings< orbit<N, T> > {
public:
    static const char *k_clazz; //!< Class name

private:
    typedef tensor_transf<N, T> tensor_transf_type;
    typedef std::pair<size_t, tensor_transf_type> pair_type;
    typedef std::map<size_t, tensor_transf_type> orbit_map_type;

public:
    typedef typename orbit_map_type::const_iterator
        iterator; //!< STL-like orbit iterator

private:
    dimensions<N> m_bidims; //!< Block index dimensions
    index<N> m_cidx; //!< Canonical index
    orbit_map_type m_orb; //!< Map of orbit indexes to transformations
    bool m_allowed; //!< Whether the orbit is allowed by symmetry

public:
    /** \brief Constructs the orbit using a symmetry group and any starter index
            in the orbit
        \param sym Symmetry group.
        \param idx Starter block index.
     **/
    orbit(const symmetry<N, T> &sym, const index<N> &idx);

    /** \brief Constructs the orbit using a symmetry group and the absolute
            value of any starter index in the orbit
        \param sym Symmetry group.
        \param aidx Absolute value of the starter block index.
     **/
    orbit(const symmetry<N, T> &sym, size_t aidx);

    /** \brief Returns whether the orbit is allowed by symmetry
     **/
    bool is_allowed() const {
        return m_allowed;
    }

    /** \brief Returns the canonical index of this orbit
     **/
    const index<N> &get_cindex() const {
        return m_cidx;
    }

    /** \brief Returns the absolute value of the canonical index of this orbit
        \deprecated Use get_acindex() instead.
     **/
    size_t get_abs_canonical_index() const {
        return m_orb.begin()->first;
    }

    /** \brief Returns the absolute value of the canonical index of this orbit
     **/
    size_t get_acindex() const {
        return m_orb.begin()->first;
    }

    /** \brief Returns the number of indexes in the orbit
     **/
    size_t get_size() const {
        return m_orb.size();
    }

    /** \brief Returns the transformation that translates the canonical block
            to a target block with the given index
        \param idx Target block index.
     **/
    const tensor_transf<N, T> &get_transf(const index<N> &idx) const;

    /** \brief Returns the transformation that translates the canonical block
            to a target block with the given index
        \param aidx Absolute value of the target block index.
     **/
    const tensor_transf<N, T> &get_transf(size_t aidx) const;

    /** \brief Checks whether the orbit contains the block with a given index
        \param idx Block index.
        \return true If the orbit contains the block.
     **/
    bool contains(const index<N> &idx) const;

    /** \brief Checks whether the orbit contains the block with the given
            absolute value of an index
        \param aidx Absolute value of a block index.
        \return true If the orbit contains the block.
     **/
    bool contains(size_t absidx) const;

    /** \brief Returns the iterator to the first (canonical) index in the orbit
     **/
    iterator begin() const {
        return m_orb.begin();
    }

    /** \brief Returns the iterator to the end of the orbit
     **/
    iterator end() const {
        return m_orb.end();
    }

    /** \brief Returns the absolute value of the index pointed at by
            the iterator
     **/
    size_t get_abs_index(const iterator &i) const;

    /** \brief Returns the transformation corresponding to the index pointed
            at by the iterator
     **/
    const tensor_transf<N, T> &get_transf(const iterator &i) const;

    //@}

private:
    void build_orbit(const symmetry<N, T> &sym, const abs_index<N> &aidx);

private:
    /** \brief Private copy constructor
     **/
    orbit(const orbit&);

};


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_H
