#ifndef LIBTENSOR_DIAG_TO_ADD_SPACE_H
#define LIBTENSOR_DIAG_TO_ADD_SPACE_H

#include <libtensor/core/noncopyable.h>
#include "diag_tensor_space.h"

namespace libtensor {


/** \brief Forms the space of the addition of two diagonal tensors
    \tparam N Tensor order.

    This algorithm takes the spaces of two input diagonal tensors and forms
    the space of the output of the addition operation.

    In terms of constraints on the tensor space, the operation produces
    the union of allowed diagonal subspaces.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_to_add_space : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    diag_tensor_space<N> m_dtsc;

public:
    /** \brief Builds the result space
     **/
    diag_to_add_space(const diag_tensor_space<N> &dtsa,
        const diag_tensor_space<N> &dtsb);

    /** \brief Returns the result space
     **/
    const diag_tensor_space<N> &get_dtsc() const {
        return m_dtsc;
    }

private:
    /** \brief Returns true if the space contains the subspace
     **/
    bool contains(const diag_tensor_space<N> &dts,
        const diag_tensor_subspace<N> &ss) const;

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TO_ADD_SPACE_H

