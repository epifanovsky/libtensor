#ifndef LIBTENSOR_DIAG_TOD_ADJUST_SPACE_H
#define LIBTENSOR_DIAG_TOD_ADJUST_SPACE_H

#include <libtensor/timings.h>
#include "diag_tensor_space.h"
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Adjust the space of a diagonal tensor without changing data
    \tparam N Tensor order.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_tod_adjust_space : public timings< diag_tod_adjust_space<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    diag_tensor_space<N> m_spc; //!< Tensor space

public:
    /** \brief Initializes the operation
        \param spc New diag space.
     **/
    diag_tod_adjust_space(const diag_tensor_space<N> &spc) : m_spc(spc) { }

    /** \brief Performs the operation
        \param ta Tensor.
     **/
    void perform(diag_tensor_wr_i<N, double> &ta);

private:
    void constrained_copy(const dimensions<N> &dims,
        const diag_tensor_subspace<N> &ss1, double *p1, size_t sz1,
        const diag_tensor_subspace<N> &ss2, double *p2, size_t sz2);

    void mark_diags(const mask<N> &m0, const diag_tensor_subspace<N> &ss,
        mask<N> &m1);

    size_t get_increment(const dimensions<N> &dims,
        const diag_tensor_subspace<N> &ss, const mask<N> &m) const;

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_ADJUST_SPACE_H

