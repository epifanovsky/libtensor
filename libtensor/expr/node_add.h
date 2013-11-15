#ifndef LIBTENSOR_EXPR_NODE_ADD_H
#define LIBTENSOR_EXPR_NODE_ADD_H

#include <map>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor addition node of the expression tree

    Node for adding expression subtrees

    \ingroup libtensor_expr
 **/
class node_add : public node {
public:
    static const char k_op_type[]; //!< Operation type

private:
    std::multimap<size_t, size_t> m_map; //!< Map

public:
    /** \brief Creates an addition node
        \param n Order of result.
     **/
    node_add(size_t n, const std::multimap<size_t, size_t> &map) :
        node(node_add::k_op_type, n), m_map(map)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_add() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_add *clone() const {
        return new node_add(*this);
    }

    const std::multimap<size_t, size_t> &get_map() const {
        return m_map;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_ADD_H
