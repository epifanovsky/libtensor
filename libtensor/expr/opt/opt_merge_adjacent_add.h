#ifndef LIBTENSOR_EXPR_OPT_MERGE_ADJACENT_ADD_H
#define LIBTENSOR_EXPR_OPT_MERGE_ADJACENT_ADD_H

#include <libtensor/expr/dag/graph.h>

namespace libtensor {
namespace expr {


/** \brief Merges adjacent addition nodes

    This optimizer locates any chained adjacent addition nodes and joins their
    arguments to produce one addition node:
    ( + ( + E1 E2 ) E3 ) --> ( + E1 E2 E3 )
    ( + E1 ( + E2 E3 ) ) --> ( + E1 E2 E3 )

    The order of arguments in additions is preserved.

    \ingroup libtensor_expr_opt
 **/
void opt_merge_adjacent_add(graph &g);


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_OPT_MERGE_ADJACENT_ADD_H
