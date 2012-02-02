#ifndef LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_IDENT_H
#define LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_IDENT_H

#include "../labeled_anytensor.h"
#include "../unassigned_expression.h"
#include "expression_node_ident.h"

namespace libtensor {


/** \brief Expression node: identity

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class unassigned_expression_node_ident :
    public unassigned_expression_node<N, T> {

private:
    labeled_anytensor<N, T> m_t; //!< Tensor

public:
    /** \brief Node constructor
     **/
    unassigned_expression_node_ident(
        labeled_anytensor<N, T> &t) :
        m_t(t) { }

    /** \brief Virtual destructor
     **/
    virtual ~unassigned_expression_node_ident() { }

    /** \brief Returns the default label
     **/
    virtual const letter_expr<N> &get_default_label() const {
        return m_t.get_label();
    }

    /** \brief Translates the expression node into the label-free form
     **/
    virtual void translate(const letter_expr<N> &label, expression<N, T> &e);

};


template<size_t N, typename T>
void unassigned_expression_node_ident<N, T>::translate(
    const letter_expr<N> &label, expression<N, T> &e) {

    permutation<N> perm(label.permutation_of(m_t.get_label()));
    expression_node_ident<N, T> n(m_t.get_tensor(), perm);
    e.add_node(n);
}


} // namespace libtensor

#endif // LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_IDENT_H
