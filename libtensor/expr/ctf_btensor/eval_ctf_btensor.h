#ifndef LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_H
#define LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_H

namespace libtensor {
namespace expr {


/** \brief Processor of evaluation plan for ctf_btensor result type
    \tparam T Tensor element type.

    \ingroup libtensor_expr_ctf_btensor
 **/
template<typename T> class eval_ctf_btensor;


} // namespace expr
} // namespace libtensor

#include "eval_ctf_btensor_double.h"

#endif // LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_H
