#ifndef LIBTENSOR_ASSIGNMENT_OPERATOR_H
#define LIBTENSOR_ASSIGNMENT_OPERATOR_H

#include "labeled_anytensor.h"
#include "expression.h"
#include "expression_dispatcher.h"
#include "ident/unassigned_expression_node_ident.h"

namespace libtensor {


/** \brief Invokes the full process of computing an expression into
        the resulting tensor

    The computation of an expression consists of the following steps:
     - Translation of the programmed expression into an expression tree,
        which removes letter labels and replaces them with a label-free
        representation. This step also includes incorporation of scaling
        factors directly into the nodes.
        (Performed in the constructor).
     - Finding the appropriate renderer for the expression depending on
        the type of the sink (left-hand side of the expression), which is
        followed by the evaluation and saving the result in the output.
        (Performed by render_and_compute()).

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class assignment_operator {
private:
    anytensor<N, T> &m_sink;
    expression<N, T> m_expr;

public:
    /** \brief Constructs the label-free representation of an expression
            using both right-hand and left-hand sides
     **/
    assignment_operator(labeled_anytensor<N, T> &lhs,
        unassigned_expression<N, T> &rhs);

    /** \brief Renders the expression into an appropriate sequence of tensor
            operations and invokes them to compute the result into the sink
     **/
    void render_and_compute();

};


template<size_t N, typename T>
assignment_operator<N, T>::assignment_operator(labeled_anytensor<N, T> &lhs,
    unassigned_expression<N, T> &rhs) : m_sink(lhs.get_tensor()) {

    rhs.translate(lhs.get_label(), m_expr);
}


template<size_t N, typename T>
void assignment_operator<N, T>::render_and_compute() {

    expression_dispatcher<N, T>::get_instance().render(m_expr, m_sink);
}


template<size_t N, typename T>
labeled_anytensor<N, T> &labeled_anytensor<N, T>::operator=(
    unassigned_expression<N, T> rhs) {

    assignment_operator<N, T>(*this, rhs).render_and_compute();
    return *this;
}


template<size_t N, typename T>
labeled_anytensor<N, T> &labeled_anytensor<N, T>::operator=(
    labeled_anytensor<N, T> rhs) {

    std::auto_ptr< unassigned_expression_node<N, T> > n(
        new unassigned_expression_node_ident<N, T>(rhs));
    unassigned_expression<N, T> e(n);
    assignment_operator<N, T>(*this, e).render_and_compute();
    return *this;
}


} // namespace libtensor

#endif // LIBTENSOR_ASSIGNMENT_OPERATOR_H
