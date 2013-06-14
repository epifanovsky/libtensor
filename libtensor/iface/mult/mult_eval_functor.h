#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_H

#include <libtensor/block_tensor/btod_mult.h>
#include "../expr/anon_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {

/** \brief Function for evaluating element-wise multiplication

     \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Recip>
class mult_eval_functor {
public:
    static const char k_clazz[]; //!< Class name

private:
    mult_core<N, T, Recip> m_core; //!< Multiply core
    interm<N, T> m_interm_a;
    interm<N, T> m_interm_b;

    btod_mult<N> *m_op; //!< Element-wise multiplication operation
    arg<N, T, oper_tag> *m_arg; //!< Composed operation argument

public:
    mult_eval_functor(
        mult_core<N, T, Recip> &core,
        const letter_expr<N> &label);

    ~mult_eval_functor();

    void evaluate();

    void clean();

    arg<N, T, oper_tag> get_arg() const { return *m_arg; }
};


template<size_t N, typename T, bool Recip>
const char mult_eval_functor<N, T, Recip>::k_clazz[] =
        "mult_eval_functor<N, T, Recip>";

template<size_t N, typename T, bool Recip>
mult_eval_functor<N, T, Recip>::mult_eval_functor(
    mult_core<N, T, Recip> &core, const letter_expr<N> &label) :

    m_core(core),
    m_interm_a(core.get_expr_1(), label),
    m_interm_b(core.get_expr_2(), label),
    m_op(0), m_arg(0) {

}


template<size_t N, typename T, bool Recip>
mult_eval_functor<N, T, Recip>::~mult_eval_functor() {

    delete m_op; m_op = 0;
    delete m_arg; m_arg = 0;
}


template<size_t N, typename T, bool Recip>
void mult_eval_functor<N, T, Recip>::evaluate() {

    m_interm_a.evaluate();
    m_interm_b.evaluate();

    if (m_arg != 0) delete m_arg;
    if (m_op != 0) delete m_op;

    arg<N, T, tensor_tag> arga = m_interm_a.get_arg();
    arg<N, T, tensor_tag> argb = m_interm_b.get_arg();

    m_op = new btod_mult<N>(
        arga.get_btensor(), arga.get_perm(),
        argb.get_btensor(), argb.get_perm(), Recip);
    m_arg = new arg<N, T, oper_tag>(*m_op,
        Recip ?  arga.get_coeff() /  argb.get_coeff() :
                arga.get_coeff() * argb.get_coeff());

}


template<size_t N, typename T, bool Recip>
void mult_eval_functor<N, T, Recip>::clean() {

    delete m_op; m_op = 0;
    delete m_arg; m_arg = 0;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_EVAL_FUNCTOR_H
