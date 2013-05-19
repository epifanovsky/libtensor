#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_XXXX_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_XXXX_H

#include <libtensor/block_tensor/btod_contract2.h>
#include "../expr/anon_eval.h"
#include "contract_core.h"
#include "contract_subexpr_labels.h"
#include "contract_contraction2_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating contractions (expression + expression case)

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class contract_eval_functor_xxxx :
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
    anon_eval<NA, T> m_eval_a; //!< Anonymous evaluator for sub-expression A
    anon_eval<NB, T> m_eval_b; //!< Anonymous evaluator for sub-expression B
    contract_contraction2_builder<N, M, K> m_contr_bld; //!< Contraction builder
    btod_contract2<N, M, K> *m_op; //!< Contraction operation
    arg<NC, T, oper_tag> *m_arg; //!< Composed operation argument

public:
    contract_eval_functor_xxxx(
        const contract_core<N, M, K, T> &core,
        const contract_subexpr_labels<N, M, K, T> &labels_ab,
        const letter_expr<NC> &label_c);

    virtual ~contract_eval_functor_xxxx();

    virtual void evaluate();

    virtual void clean();

    virtual arg<NC, T, oper_tag> get_arg() const {
        return *m_arg;
    }

private:
    void create_arg();
    void destroy_arg();

};


template<size_t N, size_t M, size_t K, typename T>
const char contract_eval_functor_xxxx<N, M, K, T>::k_clazz[] =
    "contract_eval_functor_xxxx<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
contract_eval_functor_xxxx<N, M, K, T>::contract_eval_functor_xxxx(
    const contract_core<N, M, K, T> &core,
    const contract_subexpr_labels<N, M, K, T> &labels_ab,
    const letter_expr<NC> &label_c) :

    m_eval_a(core.get_expr_1(), labels_ab.get_label_a()),
    m_eval_b(core.get_expr_2(), labels_ab.get_label_b()),
    m_contr_bld(
        labels_ab.get_label_a(), permutation<NA>(),
        labels_ab.get_label_b(), permutation<NB>(),
        label_c, core.get_contr()),
    m_op(0), m_arg(0) {

}


template<size_t N, size_t M, size_t K, typename T>
contract_eval_functor_xxxx<N, M, K, T>::~contract_eval_functor_xxxx() {

    destroy_arg();
}


template<size_t N, size_t M, size_t K, typename T>
void contract_eval_functor_xxxx<N, M, K, T>::evaluate() {

    m_eval_a.evaluate();
    m_eval_b.evaluate();
    create_arg();
}


template<size_t N, size_t M, size_t K, typename T>
void contract_eval_functor_xxxx<N, M, K, T>::clean() {

    destroy_arg();
    m_eval_a.clean();
    m_eval_b.clean();
}


template<size_t N, size_t M, size_t K, typename T>
void contract_eval_functor_xxxx<N, M, K, T>::create_arg() {

    destroy_arg();
    m_op = new btod_contract2<N, M, K>(m_contr_bld.get_contr(),
        m_eval_a.get_btensor(), m_eval_b.get_btensor());
    m_arg = new arg<NC, T, oper_tag>(*m_op, 1.0);
}


template<size_t N, size_t M, size_t K, typename T>
void contract_eval_functor_xxxx<N, M, K, T>::destroy_arg() {

    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_XXXX_H
