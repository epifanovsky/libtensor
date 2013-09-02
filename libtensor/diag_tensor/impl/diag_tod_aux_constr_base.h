#ifndef LIBTENSOR_DIAG_TOD_AUX_CONSTR_BASE_H
#define LIBTENSOR_DIAG_TOD_AUX_CONSTR_BASE_H

#include "../diag_tensor_space.h"

namespace libtensor {


template<size_t N>
class diag_tod_aux_constr_base {
protected:
    /** \brief Given a starter mask m0, produces an augmented mask m1 that has
            marked all whole diagonals that correspond to bits set in m0
     **/
    void mark_diags(
        const mask<N> &m0,
        const diag_tensor_subspace<N> &ss,
        mask<N> &m1) const;

    /** \brief Given a mask that may span multiple diagonals, produces the
            increment in an array that corresponds to the given subspace
     **/
    size_t get_increment(
        const dimensions<N> &dims,
        const diag_tensor_subspace<N> &ss,
        const mask<N> &m) const;

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_AUX_CONSTR_BASE_H
