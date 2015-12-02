#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_H

namespace libtensor {
namespace expr {


/** \brief Processor of evaluation plan for btensor result type
    \tparam T Tensor element type.

    \ingroup libtensor_expr_btensor
 **/
template<typename T> class eval_btensor;


} // namespace expr
} // namespace libtensor

#include "eval_btensor_double.h"
#include "eval_btensor_complex_double.h"

#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_H
