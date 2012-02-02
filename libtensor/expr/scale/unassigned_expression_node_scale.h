#ifndef LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_SCALE_H
#define LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_SCALE_H

#include "../unassigned_expression.h"
#include "../expression.h"

namespace libtensor {


/** \brief Expression node: scaling of a sub-expression

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class unassigned_expression_node_scale :
    public unassigned_expression_node<N, T> {

private:
    unassigned_expression<N, T> m_e; //!< Expression
    T m_s; //!< Scalar

public:
    /** \brief Node constructor
     **/
    unassigned_expression_node_scale(
        unassigned_expression<N, T> &e,
        const T &s) :
        m_e(e), m_s(s) { }

    /** \brief Virtual destructor
     **/
    virtual ~unassigned_expression_node_scale() { }

    /** \brief Returns the default label
     **/
    virtual const letter_expr<N> &get_default_label() const {
        return m_e.get_default_label();
    }

    /** \brief Translates the expression node into the label-free form
     **/
    virtual void translate(const letter_expr<N> &label,
        expression<N, T> &e);

};


template<size_t N, typename T>
void unassigned_expression_node_scale<N, T>::translate(
    const letter_expr<N> &label, expression<N, T> &e) {

    expression<N, T> e0;
    m_e.translate(label, e0);
    e0.scale(m_s);
    const std::vector< expression_node<N, T>* > &nodes = e0.get_nodes();
    for(size_t i = 0; i < nodes.size(); i++) e.add_node(*nodes[i]);
}


} // namespace libtensor

#endif // LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_SCALE_H
