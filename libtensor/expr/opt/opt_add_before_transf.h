#ifndef LIBTENSOR_EXPR_OPT_ADD_BEFORE_TRANSF_H
#define LIBTENSOR_EXPR_OPT_ADD_BEFORE_TRANSF_H

#include <libtensor/expr/dag/graph.h>

namespace libtensor {
namespace expr {


/** \brief Swaps connections to place additions before tensor transformations

    This optimizer performs the following transformation:
    ( Tr ( + E1 E2 ... En ) ) --> ( + ( Tr E1 ) ( Tr E2 ) ... ( Tr En ) )

    \ingroup libtensor_expr_opt
 **/
void opt_add_before_transf(graph &g);


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_OPT_ADD_BEFORE_TRANSF_H
