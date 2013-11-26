#ifndef LIBTENSOR_EXPR_PRINT_TREE_H
#define LIBTENSOR_EXPR_PRINT_TREE_H

#include <iostream>
#include "expr_tree.h"

namespace libtensor {
namespace expr {


/** \brief Prints part of an expr_tree
    \param tr Expression tree.
    \param h ID of head node
    \param os Output stream.
    \param indent Indentation (number of spaces on the left, default 0).

    \ingroup libtensor_expr
 **/
void print_tree(const expr_tree &tr, expr_tree::node_id_t h,
        std::ostream &os, size_t indent);


/** \brief Prints an expr_tree
    \param tr Expression tree.
    \param os Output stream.
    \param indent Indentation (number of spaces on the left, default 0).

    \ingroup libtensor_expr
 **/
inline
void print_tree(const expr_tree &tr, std::ostream &os, size_t indent = 0) {
    print_tree(tr, tr.get_root(), os, indent);
}


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_PRINT_TREE_H
