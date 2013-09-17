#ifndef LIBTENSOR_DIAG_TOD_AUX_DOTPROD_H
#define LIBTENSOR_DIAG_TOD_AUX_DOTPROD_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/core/scalar_transf_double.h>
#include "diag_tod_aux_constr_base.h"

namespace libtensor {


/** \brief Computes the dot product within the intersection of two
        diagonal subspaces
    \tparam N Tensor order

    \ingroup libtensor_diag_tensor_tod
 **/
template<size_t N>
class diag_tod_aux_dotprod :
    public diag_tod_aux_constr_base<N>,
    public timings< diag_tod_aux_dotprod<N> >, public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

private:
    dimensions<N> m_dimsa;
    dimensions<N> m_dimsb;
    const diag_tensor_subspace<N> &m_ssa;
    const diag_tensor_subspace<N> &m_ssb;
    const double *m_pa;
    const double *m_pb;
    size_t m_sza;
    size_t m_szb;
    tensor_transf<N, double> m_tra;
    tensor_transf<N, double> m_trb;

public:
    /** \brief Initializes the operation
        \param dimsa Dimensions of A.
        \param dimsb Dimensions of B.
        \param ssa Subspace in A.
        \param ssb Subspace in B.
        \param pa Data pointer in A.
        \param pb Data pointer in B.
        \param sza Length of data array in A.
        \param szb Length of data array in B.
        \param tra Transformation to be applied to A before dot product.
        \param trb Transformation to be applied to B before dot product.
     **/
    diag_tod_aux_dotprod(
        const dimensions<N> &dimsa,
        const dimensions<N> &dimsb,
        const diag_tensor_subspace<N> &ssa,
        const diag_tensor_subspace<N> &ssb,
        const double *pa,
        const double *pb,
        size_t sza,
        size_t szb,
        const tensor_transf<N, double> &tra,
        const tensor_transf<N, double> &trb) :

        m_dimsa(dimsa), m_dimsb(dimsb), m_ssa(ssa), m_ssb(ssb),
        m_pa(pa), m_pb(pb), m_sza(sza), m_szb(szb), m_tra(tra), m_trb(trb)
    { }

    /** \brief Computes the dot product
     **/
    double calculate();

protected:
    using diag_tod_aux_constr_base<N>::mark_diags;
    using diag_tod_aux_constr_base<N>::get_increment;

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_AUX_DOTPROD_H

