#ifndef LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_H
#define LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_H

#include <libtensor/dense_tensor/to_contract2_dims.h>
#include "diag_tensor_space.h"

namespace libtensor {


/** \brief Forms the space of the contraction of two diagonal tensors
    \tparam N Order of first tensor less contraction order.
    \tparam M Order of second tensor less contraction order.
    \tparam K Order of contraction.

    This algorithm takes the spaces of two input diagonal tensors and forms
    the space of the output of the contraction operation.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N, size_t M, size_t K>
class diag_to_contract2_space {
public:
    static const char *k_clazz; //!< Class name

private:
    to_contract2_dims<N, M, K> m_dimsc; //!< Dimensions of C
    diag_tensor_space<N + M> m_dtsc; //!< Space of C

public:
    /** \brief Builds the result space
     **/
    diag_to_contract2_space(const contraction2<N, M, K> &contr,
        const diag_tensor_space<N + K> &dtsa,
        const diag_tensor_space<M + K> &dtsb);

    /** \brief Returns the result space
     **/
    const diag_tensor_space<N + M> &get_dtsc() const {
        return m_dtsc;
    }

private:
    /** \brief Forms a subspace in C from subspaces A and B, adds it to
            space of C
     **/
    void add_subspace(const contraction2<N, M, K> &contr,
        const diag_tensor_subspace<N + K> &ssa,
        const diag_tensor_subspace<M + K> &ssb);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_H

