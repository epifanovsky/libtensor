#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_H

#include <memory>
#include <libtensor/exception.h>
#include "expr.h"
#include "evalfunctor.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Evaluates an expression into a %tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Core Expression core type.

    Provides the facility to evaluate an expression. This class is
    instantiated when both the expression and the recipient are known,
    and therefore all necessary %tensor operations can be constructed.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class eval : public noncopyable {
private:
    expr<N, T> m_expr; //!< Expression
    labeled_btensor<N, T, true> &m_result; //!< Result
    std::auto_ptr< eval_container_i<N, T> > m_eval_container; //!< Container

public:
    eval(const expr<N, T> &e, labeled_btensor<N, T, true> &result);

    /** \brief Evaluates the expression
     **/
    void evaluate();

};


template<size_t N, typename T>
eval<N, T>::eval(const expr<N, T> &e, labeled_btensor<N, T, true> &result) :

    m_expr(e),
    m_result(result),
    m_eval_container(m_expr.get_core().create_container(
        m_result.get_label())) {

}


template<size_t N, typename T>
void eval<N, T>::evaluate() {

    m_eval_container->prepare();
    evalfunctor<N, T>(m_expr, *m_eval_container).get_bto().
        perform(m_result.get_btensor());
    m_eval_container->clean();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_H
