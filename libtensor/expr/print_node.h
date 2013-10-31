#ifndef LIBTENSOR_EXPR_PRINT_NODE_H
#define LIBTENSOR_EXPR_PRINT_NODE_H

#include <iostream>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Prints the contents of a node
    \param n Node.
    \param os Output stream.
    \param indent Indentation (number of spaces on the left, default 0).

    \ingroup libtensor_expr
 **/
void print_node(const node &n, std::ostream &os, size_t indent = 0);


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_PRINT_NODE_H
