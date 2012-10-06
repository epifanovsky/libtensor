#ifndef LIBTENSOR_DIAG_TOD_AUX_CONSTR_ADD_H
#define LIBTENSOR_DIAG_TOD_AUX_CONSTR_ADD_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/core/scalar_transf_double.h>
#include "diag_tod_aux_constr_base.h"

namespace libtensor {


/** \brief Performs the constrained addition of two diagonal tensors in
        subspaces
    \param N Tensor order.

    This algorithm takes a subspace of a diagonal tensor A and a subspace of
    a diagonal tensor B and their respective data array pointers, as well as
    a transformation T(A). It then adds tensor elements from A to B subject
    to a constraint to the largest common subgroup of allowed elements between
    the subspaces in A and B.

    For example, if the subspace in A(ij) allows all elements, and the subspace
    in B(ij) only allows elements on the diagonal, only the diagonal elements
    from A(ij) will be added to B(ij).

    Similarly, if the subpace in A(ijk) only allows elements with i = j and the
    subspace in B(ijk) only allows elements with i = k, only those elements with
    i = j = k will be added to B(ijk).

    \ingroup libtensor_diag_tensor_tod
 **/
template<size_t N>
class diag_tod_aux_constr_add :
    public diag_tod_aux_constr_base<N>,
    public timings< diag_tod_aux_constr_add<N> >,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

private:
    dimensions<N> m_dimsa;
    const diag_tensor_subspace<N> &m_ssa;
    const double *m_pa;
    size_t m_sza;
    tensor_transf<N, double> m_tra;

public:
    /** \brief Initializes the operation
        \param dimsa Dimensions of A.
        \param ssa Subspace in A.
        \param pa Data pointer in A.
        \param sza Length of data array in A.
        \param tra Transformation to be applied to A before addition.
     **/
    diag_tod_aux_constr_add(
        const dimensions<N> &dimsa,
        const diag_tensor_subspace<N> &ssa,
        const double *pa,
        size_t sza,
        const tensor_transf<N, double> &tra) :
        m_dimsa(dimsa), m_ssa(ssa), m_pa(pa), m_sza(sza), m_tra(tra) { }

    /** \brief Performs the operation
        \param ssb Subspace in B.
        \param pb Data pointer in B.
        \param szb Length of data array in B.
     **/
    void perform(
        const diag_tensor_subspace<N> &ssb,
        double *pb,
        size_t szb);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_AUX_CONSTR_ADD_H
