#ifndef LIBTENSOR_SHORT_ORBIT_H
#define LIBTENSOR_SHORT_ORBIT_H

#include <libtensor/timings.h>
#include "abs_index.h"
#include "magic_dimensions.h"
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
    magic_dimensions<N> m_mdims; //!< Magic dimensions
    index<N> m_cidx; //!< Canonical index
    size_t m_acidx; //!< Absolute value of canonical index
    bool m_allowed; //!< Whether the orbit is allowed by symmetry

public:
    /** \brief Searches for the canonical index from any starter index in
            an orbit
        \param sym Symmetry group.
        \param idx Starter index.
        \param compute_allowed If true, compute whether the orbit is
            allowed, false (default) skips this computation
     **/
    short_orbit(const symmetry<N, T> &sym, const index<N> &idx,
        bool compute_allowed = false);

    /** \brief Searches for the canonical index from any starter index in
            an orbit
        \param sym Symmetry group.
        \param aidx Absolute value of the starter index.
        \param compute_allowed If true, compute whether the orbit is
            allowed, false (default) skips this computation
     **/
    short_orbit(const symmetry<N, T> &sym, size_t aidx,
        bool compute_allowed = false);

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
     **/
    size_t get_acindex() const {
        return m_acidx;
    }

private:
    void find_cindex(const symmetry<N, T> &sym, size_t aidx);

};


} // namespace libtensor

#endif // LIBTENSOR_SHORT_ORBIT_H
