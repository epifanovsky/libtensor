#ifndef LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_DIRSUM_H
#define LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_DIRSUM_H

#include "../unassigned_expression.h"
#include "../expression.h"
#include "expression_node_dirsum.h"

namespace libtensor {


/** \brief Expression node: direct sum of two sub-expressions
    \tparam N Order of first tensor expression.
    \tparam M Order of second tensor expression.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
class unassigned_expression_node_dirsum :
    public unassigned_expression_node<N + M, T> {

private:
    unassigned_expression<N, T> m_a; //!< First sub-expression
    unassigned_expression<M, T> m_b; //!< Second sub-expression
    letter_expr<N + M> m_label; //!< Default label

public:
    /** \brief Node constructor
     **/
    unassigned_expression_node_dirsum(
        unassigned_expression<N, T> &a,
        unassigned_expression<M, T> &b) :
        m_a(a), m_b(b),
        m_label(m_a.get_default_label(), m_b.get_default_label())
        { }

    /** \brief Virtual destructor
     **/
    virtual ~unassigned_expression_node_dirsum() { }

    /** \brief Returns the default label
     **/
    virtual const letter_expr<N + M> &get_default_label() const {
        return m_label;
    }

    /** \brief Translates the expression node into the label-free form
     **/
    virtual void translate(const letter_expr<N + M> &label,
        expression<N + M, T> &e);

};


template<size_t N, size_t M, typename T>
void unassigned_expression_node_dirsum<N, M, T>::translate(
    const letter_expr<N + M> &label, expression<N + M, T> &e) {

    expression<N, T> a;
    expression<M, T> b;
    m_a.translate(m_a.get_default_label(), a);
    m_b.translate(m_b.get_default_label(), b);
    permutation<N + M> perm(label.permutation_of(m_label));
    expression_node_dirsum<N, M, T> n(a, b, perm);
    e.add_node(n);
}


} // namespace libtensor

#endif // LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_DIRSUM_H
