#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_H

#include "../../defs.h"
#include "../../exception.h"
#include <libtensor/block_tensor/bto/direct_bto.h>
#include <libtensor/block_tensor/bto/bto_traits.h>
#include "expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T>
class evalfunctor_i {
public:
    virtual ~evalfunctor_i() { }
    virtual direct_bto< N, bto_traits<T> > &get_bto() = 0;
};


/** \brief Evaluates an expression that contains both tensors and
        operations
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Core Expression core type.
    \tparam NTensor Number of tensors in the expression.
    \tparam NOper Number of operations in the expression.

    \ingroup labeled_btensor_expr
 **/
template<size_t N, typename T, typename Core, size_t NTensor, size_t NOper>
class evalfunctor : public evalfunctor_i<N, T> {
public:
    //!    Expression type
    typedef expr<N, T, Core> expression_t;

    //!    Output labeled block %tensor type
    typedef labeled_btensor<N, T, true> result_t;

    //!    Evaluating container type
    typedef typename expression_t::eval_container_t eval_container_t;

private:
    expression_t &m_expr;
    eval_container_t &m_eval_container;

public:
    evalfunctor(expression_t &expr, eval_container_t &cont);
    virtual ~evalfunctor() { }
    virtual direct_bto<N, bto_traits<T> > &get_bto();
    virtual direct_bto<N, bto_traits<T> > &get_clean_bto();
};


} // namespace labeled_btensor_expr
} // namespace libtensor

#include "evalfunctor_double.h" // Specialization for T = double

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_H
