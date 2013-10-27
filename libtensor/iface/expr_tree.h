#ifndef LIBTENSOR_IFACE_EXPR_TREE_H
#define LIBTENSOR_IFACE_EXPR_TREE_H

#include <libtensor/expr/node.h>
#include "tensor_list.h"

namespace libtensor {
namespace iface {


/** \brief Context-free expression

    \ingroup libtensor_iface
 **/
class expr_tree {
private:
    expr::node *m_root; //!< Root node
    tensor_list m_tl; //!< List of tensors

public:
    /** \brief Constructs an expression using the root node and a database of
            tensors
     **/
    expr_tree(const expr::node &root, tensor_list &tl) :
        m_root(root.clone()), m_tl(tl, 1)
    { }

    /** \brief Copy constructor
     **/
    expr_tree(const expr_tree &e) :
        m_root(e.m_root->clone()), m_tl(e.m_tl)
    { }

    /** \brief Destructor
     **/
    ~expr_tree() {
        delete m_root;
    }

    libtensor::expr::node &get_nodes() {
        return *m_root;
    }

    const libtensor::expr::node &get_nodes() const {
        return *m_root;
    }

    tensor_list &get_tensors() {
        return m_tl;
    }

    const tensor_list &get_tensors() const {
        return m_tl;
    }
};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EXPR_TREE_H
