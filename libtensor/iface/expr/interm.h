#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_INTERM_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_INTERM_H

#include "anon_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Wrapper for computing intermediates
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class interm : public noncopyable {
private:
    anon_eval<N, T> *m_eval;
    arg<N, T, tensor_tag> *m_arg;

public:
    //!    \name Construction and destruction
    //@{

    interm(const expr_rhs<N, T> &e, const letter_expr<N> &label);

    ~interm();

    //@}

    //!    \name Evaluation
    //@{

    /** \brief Evaluates the expression
     **/
    void evaluate();

    /** \brief Cleans up the intermediates
     **/
    void clean();

    /** \brief Returns the block tensor argument
     **/
    const arg<N, T, tensor_tag> &get_arg();

    //@}

};


template<size_t N, typename T>
interm<N, T>::interm(
    const expr_rhs<N, T> &e, const letter_expr<N> &label) :

    m_eval(0), m_arg(0) {

    std::auto_ptr< eval_container_i<N, T> > cont(
        e.get_core().create_container(label));

    if(cont->get_ntensor() == 1 && cont->get_noper() == 0) {
        m_arg = new arg<N, T, tensor_tag>(cont->get_tensor_arg(0));
    } else {
        m_eval = new anon_eval<N, T>(e, label);
    }
}


template<size_t N, typename T>
interm<N, T>::~interm() {

    delete m_arg;
    delete m_eval;
}


template<size_t N, typename T>
void interm<N, T>::evaluate() {

    if(m_eval) {
        m_eval->evaluate();
        m_arg = new arg<N, T, tensor_tag>(m_eval->get_btensor(),
            permutation<N>(), T(1));
    }
}


template<size_t N, typename T>
void interm<N, T>::clean() {

    if(m_eval) {
        delete m_arg; m_arg = 0;
        m_eval->clean();
    }
}


template<size_t N, typename T>
const arg<N, T, tensor_tag> &interm<N, T>::get_arg() {

    return *m_arg;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_INTERM_H
