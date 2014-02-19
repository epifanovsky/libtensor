#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_DIRSUM_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_DIRSUM_H

#include "../eval_btensor.h"
#include "eval_btensor_evaluator_i.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


template<size_t NC>
class dirsum : public eval_btensor_evaluator_i<NC, double> {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef typename eval_btensor_evaluator_i<NC, double>::bti_traits
        bti_traits;
    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    eval_btensor_evaluator_i<NC, double> *m_impl;

public:
    /** \brief Initializes the evaluator
     **/
    dirsum(const expr_tree &tree, node_id_t &id,
        const tensor_transf<NC, double> &trc);

    /** \brief Virtual destructor
     **/
    virtual ~dirsum();

    /** \brief Returns the block tensor operation
     **/
    virtual additive_gen_bto<NC, bti_traits> &get_bto() const {
        return m_impl->get_bto();
    }

};


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_DIRSUM_H
