#ifndef LIBTENSOR_SHORT_ORBIT_H
#define LIBTENSOR_SHORT_ORBIT_H

#include <libtensor/timings.h>
#include "abs_index.h"
#include "noncopyable.h"
#include "tensor_transf.h"
#include "symmetry.h"

namespace libtensor {


/** \brief Computes the canonical index from any index of an orbit

    This is a short version of the orbit algorithm to only compute the canonical
    index from any index in the same orbit.

    \sa symmetry_i, orbit, orbit_list

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class short_orbit : public noncopyable, public timings< short_orbit<N, T> > {
public:
    static const char *k_clazz; //!< Class name

private:
    dimensions<N> m_dims; //!< Index dimensions
    index<N> m_cidx; //!< Canonical index
    size_t m_acidx; //!< Absolute value of canonical index

public:
    /** \brief Searches for the canonical index from any starter index in
            an orbit
        \param sym Symmetry group.
        \param idx Starter index.
     **/
    short_orbit(const symmetry<N, T> &sym, const index<N> &idx);

    /** \brief Searches for the canonical index from any starter index in
            an orbit
        \param sym Symmetry group.
        \param aidx Absolute value of the starter index.
     **/
    short_orbit(const symmetry<N, T> &sym, size_t aidx);

    /** \brief Returns the canonical index of this orbit
     **/
    const index<N> &get_cindex() const {
        return m_cidx;
    }

    /** \brief Returns the absolute value of the canonical index of this orbit
     **/
    size_t get_acindex() const {
        return m_acidx;
    }

private:
    void find_cindex(const symmetry<N, T> &sym, size_t aidx);

};


} // namespace libtensor

#endif // LIBTENSOR_SHORT_ORBIT_H
