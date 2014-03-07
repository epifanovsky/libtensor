#ifndef LIBTENSOR_EXPR_OPT_MERGE_EQUIV_IDENT_H
#define LIBTENSOR_EXPR_OPT_MERGE_EQUIV_IDENT_H

#include <libtensor/expr/dag/graph.h>

namespace libtensor {
namespace expr {


/** \brief Merges equivalent tensor identity leaves

    This optimizer replaces any duplicates of the identity of the same tensor
    with one identity node.

    \ingroup libtensor_expr_opt
 **/
void opt_merge_equiv_ident(graph &g);


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_OPT_MERGE_EQUIV_IDENT_H
