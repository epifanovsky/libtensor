#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_1010_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_1010_H

#include <memory>

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating contractions (tensor + tensor)

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class contract_eval_functor_1010 :
    public contract_eval_functor_base<N, M, K, T>, public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K,
        NB = M + K,
        NC = N + M
    };

private:
    std::auto_ptr< eval_container_i<NA, T> >
        m_eval_a; //!< Container for tensor A
    arg<NA, T, tensor_tag> m_arg_a; //!< Tensor argument for A
    std::auto_ptr< eval_container_i<NB, T> >
        m_eval_b; //!< Container for tensor B
    arg<NB, T, tensor_tag> m_arg_b; //!< Tensor argument for B
    contract_contraction2_builder<N, M, K> m_contr_bld; //!< Contraction builder
    btod_contract2<N, M, K> m_op; //!< Contraction operation
    arg<NC, T, oper_tag> m_arg; //!< Composed operation argument

public:
    contract_eval_functor_1010(
        const contract_core<N, M, K, T> &core,
        const contract_subexpr_labels<N, M, K, T> &labels_ab,
        const letter_expr<NC> &label_c);

    virtual ~contract_eval_functor_1010() { }

    virtual void evaluate() { }

    virtual void clean() { }

    virtual arg<N + M, T, oper_tag> get_arg() const { return m_arg; }

};


template<size_t N, size_t M, size_t K, typename T>
const char contract_eval_functor_1010<N, M, K, T>::k_clazz[] =
    "contract_eval_functor_1010<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
contract_eval_functor_1010<N, M, K, T>::contract_eval_functor_1010(
    const contract_core<N, M, K, T> &core,
    const contract_subexpr_labels<N, M, K, T> &labels_ab,
    const letter_expr<NC> &label_c) :

    m_eval_a(core.get_expr_1().get_core().create_container(
        labels_ab.get_label_a())),
    m_arg_a(m_eval_a->get_tensor_arg(0)),
    m_eval_b(core.get_expr_2().get_core().create_container(
        labels_ab.get_label_b())),
    m_arg_b(m_eval_b->get_tensor_arg(0)),
    m_contr_bld(
        labels_ab.get_label_a(), permutation<NA>(m_arg_a.get_perm(), true),
        labels_ab.get_label_b(), permutation<NB>(m_arg_b.get_perm(), true),
        label_c, core.get_contr()),
    m_op(m_contr_bld.get_contr(), m_arg_a.get_btensor(), m_arg_b.get_btensor()),
    m_arg(m_op, m_arg_a.get_coeff() * m_arg_b.get_coeff()) {

}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_1010_H
