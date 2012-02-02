#ifndef LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_DIAG_H
#define LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_DIAG_H

#include "../unassigned_expression.h"
#include "../expression.h"
#include "expression_node_diag.h"

namespace libtensor {


/** \brief Expression node: general diagonal of a sub-expression
    \tparam N Order of tensor expression.
    \tparam M Order of diagonal.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
class unassigned_expression_node_diag :
    public unassigned_expression_node<N - M + 1, T> {

private:
    unassigned_expression<N, T> m_a; //!< Sub-expression
    letter_expr<M> m_diag; //!< Diagonal label
    const letter &m_let; //!< Diagonal letter on output
    letter_expr<N - M + 1> m_label; //!< Default label

public:
    /** \brief Node constructor
     **/
    unassigned_expression_node_diag(
        unassigned_expression<N, T> &a,
        const letter_expr<M> &diag, const letter &let) :
        m_a(a), m_diag(diag), m_let(let),
        m_label(make_default_label(m_a.get_default_label(), m_diag, m_let))
        { }

    /** \brief Virtual destructor
     **/
    virtual ~unassigned_expression_node_diag() { }

    /** \brief Returns the default label
     **/
    virtual const letter_expr<N - M + 1> &get_default_label() const {
        return m_label;
    }

    /** \brief Translates the expression node into the label-free form
     **/
    virtual void translate(const letter_expr<N - M + 1> &label,
        expression<N - M + 1, T> &e);

private:
    static letter_expr<N - M + 1> make_default_label(const letter_expr<N> &la,
        const letter_expr<M> &ld, const letter &l);

};


template<size_t N, size_t M, typename T>
void unassigned_expression_node_diag<N, M, T>::translate(
    const letter_expr<N - M + 1> &label, expression<N - M + 1, T> &e) {

    expression<N, T> a;
    m_a.translate(m_a.get_default_label(), a);
    permutation<N - M + 1> perm(label.permutation_of(m_label));
    mask<N> msk;
    expression_node_diag<N, M, T> n(a, msk, perm);
    e.add_node(n);
}


template<size_t N, size_t M, typename T>
letter_expr<N - M + 1>
unassigned_expression_node_diag<N, M, T>::make_default_label(
    const letter_expr<N> &la, const letter_expr<M> &ld, const letter &l) {

    sequence<N - M + 1, const letter*> seq(0);
    for(size_t i = 0; i < M; i++) if(!la.contains(ld.letter_at(i))) throw 0;
    size_t j = 0;
    for(size_t i = 0; i < N && j < N - M + 1; i++) {
        if(!ld.contains(la.letter_at(i))) seq[j++] = &la.letter_at(i);
    }
    if(j != N - M) throw 0;
    seq[j++] = &l;
    return letter_expr<N - M + 1>(seq);
}


} // namespace libtensor

#endif // LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_DIAG_H
