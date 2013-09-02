#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_XX_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_XX_H

#include <libtensor/block_tensor/btod_diag.h>
#include "../expr/direct_eval.h"
#include "diag_core.h"
#include "diag_subexpr_label_builder.h"
#include "diag_params_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating the diagonal

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class diag_eval_functor_xx : public diag_eval_functor_base<N, M, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    direct_eval<N, T> m_eval_a; //!< Direct evaluator for the sub-expression
    diag_params_builder<N, M> m_params_bld; //!< Parameters builder
    btod_diag<N, M> *m_op; //!< Diagonal extraction operation
    arg<N - M + 1, T, oper_tag> *m_arg; //!< Composed operation argument

public:
    diag_eval_functor_xx(
        const diag_core<N, M, T> &core,
        const diag_subexpr_label_builder<N, M> &labels_a,
        const letter_expr<N - M + 1> &label_b);

    virtual ~diag_eval_functor_xx();

    virtual void evaluate();

    virtual void clean();

    virtual arg<N - M + 1, T, oper_tag> get_arg() const {
        return *m_arg;
    }

private:
    void create_arg();
    void destroy_arg();

};


template<size_t N, size_t M, typename T>
const char diag_eval_functor_xx<N, M, T>::k_clazz[] = "diag_eval_functor_xx<N, M, T>";


template<size_t N, size_t M, typename T>
diag_eval_functor_xx<N, M, T>::diag_eval_functor_xx(
    const diag_core<N, M, T> &core,
    const diag_subexpr_label_builder<N, M> &label_a,
    const letter_expr<N - M + 1> &label_b) :

    m_eval_a(core.get_sub_expr(), label_a.get_label()),
    m_params_bld(
        label_a.get_label(), permutation<N>(), label_b,
        core.get_diag_letter(), core.get_diag_label()),
    m_op(0), m_arg(0) {

}


template<size_t N, size_t M, typename T>
diag_eval_functor_xx<N, M, T>::~diag_eval_functor_xx() {

    destroy_arg();
}


template<size_t N, size_t M, typename T>
void diag_eval_functor_xx<N, M, T>::evaluate() {

    m_eval_a.evaluate();
    create_arg();
}


template<size_t N, size_t M, typename T>
void diag_eval_functor_xx<N, M, T>::clean() {

    destroy_arg();
    m_eval_a.clean();
}


template<size_t N, size_t M, typename T>
void diag_eval_functor_xx<N, M, T>::create_arg() {

    destroy_arg();
    m_op = new btod_diag<N, M>(m_eval_a.get_btensor(),
        m_params_bld.get_mask(), m_params_bld.get_perm());
    m_arg = new arg<N - M + 1, T, oper_tag>(*m_op, 1.0);
}


template<size_t N, size_t M, typename T>
void diag_eval_functor_xx<N, M, T>::destroy_arg() {

    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_XX_H
