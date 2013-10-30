#ifndef LIBTENSOR_EXPR_NODE_IDENT_H
#define LIBTENSOR_EXPR_NODE_IDENT_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor identity node of the expression tree

    \ingroup libtensor_expr
 **/
class node_ident : public node {
private:
    unsigned m_tid; //!< Tensor ID

public:
    /** \brief Creates an identity node
        \param tid Tensor ID.
     **/
    node_ident(unsigned tid) :
        node("ident"), m_tid(tid)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_ident() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_ident *clone() const {
        return new node_ident(*this);
    }

    /** \brief Applies a map to argument IDs
     **/
//    virtual void apply_id_map(const std::map<unsigned, unsigned> &m) {
//        m_tid = m[m_tid];
//    }

    /** \brief Returns tensor ID
     **/
    unsigned get_tid() const {
        return m_tid;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_IDENT_H
