#ifndef LIBTENSOR_UNASSIGNED_EXPRESSION_H
#define LIBTENSOR_UNASSIGNED_EXPRESSION_H

#include <memory>
#include "letter_expr.h"

namespace libtensor {


template<size_t N, typename T>
class expression;


/** \brief Container for a node in tensor expressions

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class unassigned_expression_node {
public:
    /** \brief Virtual destructor
     **/
    virtual ~unassigned_expression_node() { }

    /** \brief Returns the default label of the output that can be used in
            translate()
     **/
    virtual const letter_expr<N> &get_default_label() const = 0;

    /** \brief Translates the expression node into the label-free form
        \param label Letter label of the result.
        \param[out] e Label-free expression.
     **/
    virtual void translate(const letter_expr<N> &label,
        expression<N, T> &e) = 0;

};


/** \brief Container for the right-hand side of tensor expressions

    This container the right-hand side of a tensor expression exactly as
    provided in the equation. Because it contains no information about
    the recipient of the expression (the left-hand side of the expression),
    it operates in terms of labeled_anytensor objects.

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class unassigned_expression {
private:
    typedef unassigned_expression_node<N, T> node_type;

private:
    mutable std::auto_ptr<node_type> m_root; //!< Root node

public:
    /** \brief Initializes the expression
        \param root Auto-pointer to the root node.
     **/
    unassigned_expression(std::auto_ptr<node_type> root) :
        m_root(root) { }

    /** \brief Copy constructor (steals the root node)
     **/
    unassigned_expression(const unassigned_expression<N, T> &other) :
        m_root(other.m_root) { }

    /** \brief Returns the default label of the output that can be used in
            translate()
     **/
    const letter_expr<N> &get_default_label() const {
        return m_root->get_default_label();
    }

    /** \brief Using the label of the output, translates the expression
            into the label-free form
     **/
    void translate(const letter_expr<N> &label, expression<N, T> &e) {
        m_root->translate(label, e);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_UNASSIGNED_EXPRESSION_H
