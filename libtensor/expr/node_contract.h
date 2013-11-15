#ifndef LIBTENSOR_EXPR_NODE_CONTRACT_H
#define LIBTENSOR_EXPR_NODE_CONTRACT_H

#include <map>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor contraction node of the expression tree

    Represents the generalized contraction of n subexpression over indexes
    described by the map. Assuming the tensor indexes are arranged
    successively starting with the indexes of the first tensor argument
    the map connects the index pairs of the tensors (key-value pairs) which
    should be contracted. The additional contraction flag marks, if
    summation over those index pairs should be carried out. Reordering of
    result indexes can be achieved by a subsequent transformation node.

    For example, the contraction of three tensors
    \f$ \sum_{rs} A_{pr} B_{rs} C_{sq} \f$
    would be represented by the contraction map
    \code { {1,2},{3,4} } \endcode

    \ingroup libtensor_expr
 **/
class node_contract : public node {
public:
    static const char k_op_type[]; //!< Operation type

private:
    std::multimap<size_t, size_t> m_map; //!< Map
    bool m_do_contr; //!< Perform contraction

public:
    /** \brief Creates a contraction node of two tensors
        \param n Order of result
        \param map Contraction map
        \param do_contr Perform summation
     **/
    node_contract(
        size_t n,
        const std::multimap<size_t, size_t> &map,
        bool do_contr = true) :
        node(node_contract::k_op_type, n), m_map(map), m_do_contr(do_contr)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_contract() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_contract *clone() const {
        return new node_contract(*this);
    }

    const std::multimap<size_t, size_t> &get_map() const {
        return m_map;
    }

    bool do_contract() const {
        return m_do_contr;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_CONTRACT_H
