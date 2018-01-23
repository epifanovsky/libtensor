#ifndef LIBTENSOR_EXPR_OPT_MERGE_ADJACENT_TRANSF_H
#define LIBTENSOR_EXPR_OPT_MERGE_ADJACENT_TRANSF_H

#include <libtensor/expr/dag/graph.h>

namespace libtensor {
namespace expr {


/** \brief Merges adjacent transformation nodes

    This optimizer locates any chained adjacent transformation nodes and
    replaces them with one representing the combined transformation:
    ( Tr[2] ( Tr[1] E ) ) --> ( Tr[1+2] E )

    \ingroup libtensor_expr_opt
 **/
template<typename T>
void opt_merge_adjacent_transf(graph &g);


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_OPT_MERGE_ADJACENT_TRANSF_H
