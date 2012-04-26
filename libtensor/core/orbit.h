#ifndef LIBTENSOR_ORBIT_H
#define LIBTENSOR_ORBIT_H

#include <map>
#include <utility>
#include <vector>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "abs_index.h"
#include "tensor_transf.h"
#include "symmetry.h"

namespace libtensor {


/** \brief Symmetry-equivalent blocks of a block %tensor

    The action of the %index %symmetry group on the set of all block
    indexes in a block %tensor generates an %index set partition, each
    subset being an orbit. The smallest %index in an orbit is its canonical
    %index. The block %tensor shall only keep the block that corresponds
    to the canonical %index. All the blocks that are connected with the
    canonical block via %symmetry elements can be obtained by applying
    a transformation to the canonical block.

    <b>Orbit evaluation</b>
    <b>Orbit iterator</b>

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class orbit : public timings<orbit<N, T> > {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename std::map< size_t, tensor_transf<N, T> >::const_iterator
        iterator; //!< Orbit iterator

private:
    typedef std::pair< size_t, tensor_transf<N, T> > pair_t;
    typedef std::map< size_t, tensor_transf<N, T> > orbit_map_t;

private:
    dimensions<N> m_bidims; //!< Block %index %dimensions
    orbit_map_t m_orb; //!< Map of %orbit indexes to transformations
    bool m_allowed; //!< Whether the orbit is allowed by %symmetry

public:
    /** \brief Constructs the %orbit using a %symmetry group and
            any %index in the %orbit
     **/
    orbit(const symmetry<N, T> &sym, const index<N> &idx);

    /** \brief Returns whether the %orbit is allowed by %symmetry
     **/
    bool is_allowed() const {

        return m_allowed;
    }

    /** \brief Returns the canonical %index of this %orbit
     **/
    size_t get_abs_canonical_index() const {

        return m_orb.begin()->first;
    }

    /** \brief Returns the number of indexes in the orbit
     **/
    size_t get_size() const {

        return m_orb.size();
    }

    /** \brief Obtain transformation of canonical block to yield block at idx.
        @param idx Block index
        @return Transformation to obtain the block at idx from the canonical block
     **/
    const tensor_transf<N, T> &get_transf(const index<N> &idx) const;

    /** \brief Obtain transformation of canonical block to yield block at absidx.
        @param absidx Absolute block index
        @return Transformation to yield block at absidx
     **/
    const tensor_transf<N, T> &get_transf(size_t absidx) const;

    /** \brief Checks if orbit contains block at idx.
        @param idx Block index
        @return True if orbit contains the block
     **/
    bool contains(const index<N> &idx) const;

    /** \brief Checks if orbit contains block at absidx.
        @param absidx Absolute block index
        @return True if orbit contains the block
     **/
    bool contains(size_t absidx) const;

    //!    \name STL-like %orbit iterator
    //@{

    iterator begin() const {

        return m_orb.begin();
    }

    iterator end() const {

        return m_orb.end();
    }

    size_t get_abs_index(iterator &i) const;

    const tensor_transf<N, T> &get_transf(iterator &i) const;

    //@}

private:
    void build_orbit(const symmetry<N, T> &sym, const index<N> &idx);

};


} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class orbit<1, double>;
    extern template class orbit<2, double>;
    extern template class orbit<3, double>;
    extern template class orbit<4, double>;
    extern template class orbit<5, double>;
    extern template class orbit<6, double>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "orbit_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_ORBIT_H
