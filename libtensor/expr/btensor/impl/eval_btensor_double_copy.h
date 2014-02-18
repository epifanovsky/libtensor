#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_COPY_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_COPY_H

#include "../eval_btensor.h"
#include "eval_btensor_evaluator_i.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


template<size_t N>
class copy : public eval_btensor_evaluator_i<N, double> {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef typename eval_btensor_evaluator_i<N, double>::bti_traits bti_traits;
    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    eval_btensor_evaluator_i<N, double> *m_impl;
    bool m_add;

public:
    /** \brief Initializes the evaluator
     **/
    copy(const expr_tree &tree, node_id_t &id,
        const tensor_transf<N, double> &tr, bool add);

    /** \brief Virtual destructor
     **/
    virtual ~copy();

    /** \brief Returns the block tensor operation
     **/
    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return m_impl->get_bto();
    }

    /** \brief Evaluates the result into given node
     **/
    void evaluate(const node &t);

};


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_COPY_H
