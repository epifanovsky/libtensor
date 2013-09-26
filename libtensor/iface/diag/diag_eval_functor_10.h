#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_10_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_10_H

#include <memory>

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating the diagonal (tensor)

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class diag_eval_functor_10 : public diag_eval_functor_base<N, M, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
//    std::auto_ptr< eval_container_i<N, T> >
//        m_eval_a; //!< Container for tensor A
//    arg<N, T, tensor_tag> m_arg_a; //!< Tensor argument for A
//    diag_params_builder<N, M> m_params_bld; //!< Parameters builder
//    btod_diag<N, M> m_op; //!< Diagonal extraction operation
//    arg<N - M + 1, T, oper_tag> m_arg; //!< Composed operation argument

public:
    diag_eval_functor_10(
        const diag_core<N, M, T> &core,
        const diag_subexpr_label_builder<N, M> &label_a,
        const letter_expr<N - M + 1> &label_b);

    virtual ~diag_eval_functor_10() { }

    virtual void evaluate() { }

    virtual void clean() { }

//    virtual arg<N - M + 1, T, oper_tag> get_arg() const {
//        return m_arg;
//    }

};


template<size_t N, size_t M, typename T>
const char diag_eval_functor_10<N, M, T>::k_clazz[] =
    "diag_eval_functor_10<N, M, T>";


template<size_t N, size_t M, typename T>
diag_eval_functor_10<N, M, T>::diag_eval_functor_10(
    const diag_core<N, M, T> &core,
    const diag_subexpr_label_builder<N, M> &label_a,
    const letter_expr<N - M + 1> &label_b) /*:

    m_eval_a(core.get_sub_expr().get_core().create_container(label_a.get_label())),
    m_arg_a(m_eval_a->get_tensor_arg(0)),
    m_params_bld(
        label_a.get_label(), permutation<N>(m_arg_a.get_perm(), true),
        label_b, core.get_diag_letter(), core.get_diag_label()),
    m_op(m_arg_a.get_btensor(), m_params_bld.get_mask(),
        m_params_bld.get_perm(), m_arg_a.get_coeff()),
    m_arg(m_op, 1.0)*/ {

}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_10_H
