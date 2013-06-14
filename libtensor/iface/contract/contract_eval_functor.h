#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_H

#include <libtensor/block_tensor/btod_contract2.h>
#include "../expr/interm.h"
#include "contract_subexpr_labels.h"
#include "contract_contraction2_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating contractions (base class)

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class contract_eval_functor_base {
public:
    enum {
        NA = N + K,
        NB = M + K,
        NC = N + M
    };

public:
    virtual ~contract_eval_functor_base() { }
    virtual void evaluate() = 0;
    virtual void clean() = 0;
    virtual arg<NC, T, oper_tag> get_arg() const = 0;

};


/** \brief Functor for evaluating contractions (top-level wrapper)

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class contract_eval_functor {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K,
        NB = M + K,
        NC = N + M
    };

private:
//    contract_eval_functor_base<N, M, K, T> *m_func;
    contract_core<N, M, K, T> &m_core;
    letter_expr<NA> m_label_a;
    letter_expr<NB> m_label_b;
    letter_expr<NC> m_label_c;
    interm<NA, T> m_interm_a;
    interm<NB, T> m_interm_b;
//    contract_contraction2_builder<N, M, K> m_contr_bld; //!< Contraction builder
    btod_contract2<N, M, K> *m_op; //!< Contraction operation
    arg<NC, T, oper_tag> *m_arg; //!< Composed operation argument

public:
    contract_eval_functor(
        contract_core<N, M, K, T> &core,
        const contract_subexpr_labels<N, M, K, T> &labels_ab,
        const letter_expr<NC> &label_c);

    ~contract_eval_functor();

    void evaluate();// {
//        m_func->evaluate();
//    }

    void clean() {
//        m_func->clean();
    }

    arg<NC, T, oper_tag> get_arg() const {
        return *m_arg;
//        return m_func->get_arg();
    }

};


} // namespace labeled_btensor_expr
} // namespace libtensor

//#include "contract_eval_functor_xxxx.h"
//#include "contract_eval_functor_xx10.h"
//#include "contract_eval_functor_10xx.h"
//#include "contract_eval_functor_1010.h"

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, size_t K, typename T>
const char contract_eval_functor<N, M, K, T>::k_clazz[] =
    "contract_eval_functor<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
contract_eval_functor<N, M, K, T>::contract_eval_functor(
    contract_core<N, M, K, T> &core,
    const contract_subexpr_labels<N, M, K, T> &labels_ab,
    const letter_expr<NC> &label_c) :

    m_core(core),
    m_label_a(labels_ab.get_label_a()),
    m_label_b(labels_ab.get_label_b()),
    m_label_c(label_c),
    m_interm_a(core.get_expr_1(), m_label_a),
    m_interm_b(core.get_expr_2(), m_label_b),
    m_op(0), m_arg(0) {

}


template<size_t N, size_t M, size_t K, typename T>
contract_eval_functor<N, M, K, T>::~contract_eval_functor() {

    delete m_op;
    delete m_arg;
}


template<size_t N, size_t M, size_t K, typename T>
void contract_eval_functor<N, M, K, T>::evaluate() {

    m_interm_a.evaluate();
    m_interm_b.evaluate();

    arg<NA, T, tensor_tag> arga = m_interm_a.get_arg();
    arg<NB, T, tensor_tag> argb = m_interm_b.get_arg();

    contract_contraction2_builder<N, M, K> cb(
        m_label_a, permutation<NA>(arga.get_perm(), true),
        m_label_b, permutation<NB>(argb.get_perm(), true),
        m_label_c, m_core.get_contr());

    m_op = new btod_contract2<N, M, K>(cb.get_contr(),
        arga.get_btensor(), argb.get_btensor());
    m_arg = new arg<NC, T, oper_tag>(*m_op,
        arga.get_coeff() * argb.get_coeff());
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_H
