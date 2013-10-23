#ifndef LIBTENSOR_IFACE_EXPR_H
#define LIBTENSOR_IFACE_EXPR_H

#include <libtensor/expr/node.h>
#include "tensor_list.h"

namespace libtensor {
namespace iface {


/** \brief Context-free expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class expr {
private:
    expr::node *m_root; //!< Root node
    tensor_list m_tl; //!< List of tensors

public:
    /** \brief Constructs an expression using the root node and a database of
            tensors
     **/
    expr(const expr::node &root, tensor_list &tl) :
        m_root(root.clone()), m_tl(tl, 1)
    { }

    /** \brief Copy constructor
     **/
    expr(const expr &e) :
        m_root(e.m_root->clone()), m_tl(e.m_tl)
    { }

    /** \brief Destructor
     **/
    ~expr() {
        delete m_root;
    }

};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EXPR_H
