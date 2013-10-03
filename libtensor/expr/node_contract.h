#ifndef LIBTENSOR_EXPR_NODE_CONTRACT_H
#define LIBTENSOR_EXPR_NODE_CONTRACT_H

#include <map>
#include "binary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor contraction node of the expression tree

    Represents the contraction of two subexpressions over indexes described
    by the contraction map. The contraction map connects the contraction
    indexes of the left tensor (keys) with the respective indexes of the right
    tensor (values).

    The index order of the result is assumed to be the external
    (non-contracted) indexes of the left tensor followed by those of the
    right tensor. Reordering of result indexes can be achieved by a subsequent
    transformation node.

    \ingroup libtensor_expr
 **/
class node_contract : public binary_node_base {
private:
    std::map<size_t, size_t> m_contr; //!< Contraction map

public:
    /** \brief Creates an contraction node
        \param contr Contraction map
     **/
    node_contract(const node &left, const node &right,
        const std::map<size_t, size_t> &contr) :
        binary_node_base("contract", left, right), m_contr(contr)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_contract() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_contract *clone() const {
        return new node_contract(*this);
    }

    const std::map<size_t, size_t> &get_contraction() const {
        return m_contr;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_CONTRACT_H
