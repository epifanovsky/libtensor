#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_AUTOSELECT_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_AUTOSELECT_H

#include "../eval_btensor.h"
#include "eval_btensor_evaluator_i.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


template<size_t N, typename T>
class autoselect : public eval_btensor_evaluator_i<N, T> {
public:
    enum {
        Nmax = eval_btensor<T>::Nmax
    };

    typedef typename eval_btensor_evaluator_i<N, T>::bti_traits bti_traits;
    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    const expr_tree &m_tree;
    eval_btensor_evaluator_i<N, T> *m_impl;

public:
    /** \brief Initializes the evaluator
     **/
    autoselect(const expr_tree &tree, node_id_t &id,
        const tensor_transf<N, T> &tr);

    /** \brief Virtual destructor
     **/
    virtual ~autoselect();

    /** \brief Returns the block tensor operation
     **/
    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return m_impl->get_bto();
    }

    /** \brief Evaluates the result into given node
     **/
    void evaluate(node_id_t lhs, bool add);

};


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_AUTOSELECT_H
