#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_H

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Evaluates an expression that contains both tensors and operations
        (interface)
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup labeled_btensor_expr
 **/
template<size_t N, typename T>
class evalfunctor_i;


/** \brief Evaluates an expression that contains both tensors and operations
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup labeled_btensor_expr
 **/
template<size_t N, typename T>
class evalfunctor;


} // namespace labeled_btensor_expr
} // namespace libtensor

#include "evalfunctor_double.h" // Specialization for T = double

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_H
