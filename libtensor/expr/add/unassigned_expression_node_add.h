#ifndef LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_ADD_H
#define LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_ADD_H

#include "../unassigned_expression.h"
#include "../expression.h"

namespace libtensor {


/** \brief Expression node: addition of two sub-expressions

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class unassigned_expression_node_add :
    public unassigned_expression_node<N, T> {

private:
    unassigned_expression<N, T> m_lhs; //!< Left-hand side
    unassigned_expression<N, T> m_rhs; //!< Right-hand side

public:
    /** \brief Node constructor
     **/
    unassigned_expression_node_add(
        unassigned_expression<N, T> &lhs,
        unassigned_expression<N, T> &rhs) :
        m_lhs(lhs), m_rhs(rhs) { }

    /** \brief Virtual destructor
     **/
    virtual ~unassigned_expression_node_add() { }

    /** \brief Returns the default label
     **/
    virtual const letter_expr<N> &get_default_label() const {
        return m_lhs.get_default_label();
    }

    /** \brief Translates the expression node into the label-free form
     **/
    virtual void translate(const letter_expr<N> &label,
        expression<N, T> &e);

};


template<size_t N, typename T>
void unassigned_expression_node_add<N, T>::translate(
    const letter_expr<N> &label, expression<N, T> &e) {

    expression<N, T> lhs, rhs;
    m_lhs.translate(label, lhs);
    m_rhs.translate(label, rhs);
    const std::vector< expression_node<N, T>* > &lhs_nodes = lhs.get_nodes();
    const std::vector< expression_node<N, T>* > &rhs_nodes = rhs.get_nodes();
    for(size_t i = 0; i < lhs_nodes.size(); i++) e.add_node(*lhs_nodes[i]);
    for(size_t i = 0; i < rhs_nodes.size(); i++) e.add_node(*rhs_nodes[i]);
}


} // namespace libtensor

#endif // LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_ADD_H
