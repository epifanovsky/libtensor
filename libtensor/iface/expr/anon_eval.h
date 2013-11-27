#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ANON_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ANON_EVAL_H

#include <memory>
#include <libtensor/exception.h>
#include <libtensor/core/noncopyable.h>
#include "../btensor.h"
#include "expr.h"
#include "evalfunctor.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Evaluates an expression into an anonymous %tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class anon_eval : public noncopyable {
private:
    expr<N, T> m_expr; //!< Expression
    std::auto_ptr< eval_container_i<N, T> > m_eval_container; //!< Container
    evalfunctor<N, T> m_functor; //!< Evaluation functor
    btensor<N, T> *m_bt; //!< Block tensor

public:
    //!    \name Construction and destruction
    //@{

    anon_eval(const expr<N, T> &e, const letter_expr<N> &label);

    ~anon_eval();

    //@}

    //!    \name Evaluation
    //@{

    /** \brief Evaluates the expression
     **/
    void evaluate();

    /** \brief Cleans up the temporary block %tensor
     **/
    void clean();

    /** \brief Returns the block %tensor
     **/
    btensor_i<N, T> &get_btensor() {
        return *m_bt;
    }

    //@}

};


template<size_t N, typename T>
anon_eval<N, T>::anon_eval(
    const expr<N, T> &e, const letter_expr<N> &label) :

    m_expr(e), m_eval_container(m_expr.get_core().create_container(label)),
    m_functor(m_expr, *m_eval_container), m_bt(0) {

}


template<size_t N, typename T>
anon_eval<N, T>::~anon_eval() {

    clean();
}


template<size_t N, typename T>
void anon_eval<N, T>::evaluate() {

    delete m_bt;

    m_eval_container->prepare();
    m_bt = new btensor<N, T>(m_functor.get_bto().get_bis());
    m_functor.get_bto().perform(*m_bt);
    m_eval_container->clean();
}


template<size_t N, typename T>
void anon_eval<N, T>::clean() {

    delete m_bt; m_bt = 0;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ANON_EVAL_H
