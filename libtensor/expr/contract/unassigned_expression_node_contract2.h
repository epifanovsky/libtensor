#ifndef LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_CONTRACT2_H
#define LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_CONTRACT2_H

#include "../unassigned_expression.h"
#include "../expression.h"
#include "expression_node_contract2.h"

namespace libtensor {


/** \brief Expression node: contraction of two sub-expressions
    \tparam N Order of first tensor expression less contracted indexes.
    \tparam M Order of second tensor expression less contracted indexes.
    \tparam K Number of contracted indexes.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class unassigned_expression_node_contract2 :
    public unassigned_expression_node<N + M, T> {

private:
    letter_expr<K> m_contr; //!< Contracted indexes
    unassigned_expression<N + K, T> m_a; //!< First sub-expression
    unassigned_expression<M + K, T> m_b; //!< Second sub-expression
    letter_expr<N + M> m_label; //!< Default label

public:
    /** \brief Node constructor
     **/
    unassigned_expression_node_contract2(
        const letter_expr<K> &contr,
        unassigned_expression<N + K, T> &a,
        unassigned_expression<M + K, T> &b) :
        m_contr(contr), m_a(a), m_b(b),
        m_label(make_default_label(m_contr, m_a, m_b)) { }

    /** \brief Virtual destructor
     **/
    virtual ~unassigned_expression_node_contract2() { }

    /** \brief Returns the default label
     **/
    virtual const letter_expr<N + M> &get_default_label() const {
        return m_label;
    }

    /** \brief Translates the expression node into the label-free form
     **/
    virtual void translate(const letter_expr<N + M> &label,
        expression<N + M, T> &e);

private:
    static letter_expr<N + M> make_default_label(const letter_expr<K> &contr,
        unassigned_expression<N + K, T> &a, unassigned_expression<M + K, T> &b);

};


template<size_t N, size_t M, size_t K, typename T>
void unassigned_expression_node_contract2<N, M, K, T>::translate(
    const letter_expr<N + M> &label, expression<N + M, T> &e) {

    expression<N + K, T> a;
    expression<M + K, T> b;
    const letter_expr<N + K> &label_a = m_a.get_default_label();
    const letter_expr<M + K> &label_b = m_b.get_default_label();
    m_a.translate(label_a, a);
    m_b.translate(label_b, b);
    contraction2<N, M, K> contr(label.permutation_of(m_label));
    for(size_t i = 0; i < K; i++) {
        size_t ia = label_a.index_of(m_contr.letter_at(i));
        size_t ib = label_b.index_of(m_contr.letter_at(i));
        contr.contract(ia, ib);
    }

    expression_node_contract2<N, M, K, T> n(contr, a, b);
    e.add_node(n);
}


template<size_t N, size_t M, size_t K, typename T>
letter_expr<N + M>
unassigned_expression_node_contract2<N, M, K, T>::make_default_label(
    const letter_expr<K> &contr, unassigned_expression<N + K, T> &a,
    unassigned_expression<M + K, T> &b) {

    sequence<N + M, const letter*> seq(0);
    size_t iseq = 0;
    const letter_expr<N + K> &label_a = a.get_default_label();
    for(size_t i = 0; i < N + K; i++) {
        if(!contr.contains(label_a.letter_at(i))) {
            seq[iseq++] = &label_a.letter_at(i);
        }
    }
    const letter_expr<M + K> &label_b = b.get_default_label();
    for(size_t i = 0; i < M + K; i++) {
        if(!contr.contains(label_b.letter_at(i))) {
            seq[iseq++] = &label_b.letter_at(i);
        }
    }
    return letter_expr<N + M>(seq);
}


} // namespace libtensor

#endif // LIBTENSOR_UNASSIGNED_EXPRESSION_NODE_CONTRACT2_H
