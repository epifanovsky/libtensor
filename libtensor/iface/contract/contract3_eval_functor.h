#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT3_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT3_EVAL_FUNCTOR_H
#if 0
#include <libtensor/block_tensor/btod_contract3.h>
#include "../expr/interm.h"
#include "contract_subexpr_labels.h"
#include "contract_contraction2_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating three-tensor contractions (top-level wrapper)

    \ingroup libtensor_btensor_expr
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
class contract3_eval_functor {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        K2 = K2a + K2b,
        NA = N1 + K1 + K2a,
        NB = N2 + K1 + K2b,
        NC = N3 + K2,
        ND = N1 + N2 + N3
    };

private:
    contract3_core<N1, N2, N3, K1, K2a, K2b, T> &m_core;
    letter_expr<NA> m_label_a;
    letter_expr<NB> m_label_b;
    letter_expr<NC> m_label_c;
    letter_expr<ND> m_label_d;
    interm<NA, T> m_interm_a;
    interm<NB, T> m_interm_b;
    interm<NC, T> m_interm_c;
    btod_contract3<N1 + K2a, N2 + K2b - K2a, N3, K1, K2> *m_op; //!< Contraction operation
    arg<ND, T, oper_tag> *m_arg; //!< Composed operation argument

public:
    contract3_eval_functor(
        contract3_core<N1, N2, N3, K1, K2a, K2b, T> &core,
        const contract_subexpr_labels<N, M, K, T> &labels_ab,
        const letter_expr<ND> &label_d);

    ~contract3_eval_functor();

    void evaluate();

    void clean();

    arg<ND, T, oper_tag> get_arg() const {
        return *m_arg;
    }

};


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
const char contract3_eval_functor<N1, N2, N3, K1, K2a, K2b, T>::k_clazz[] =
    "contract3_eval_functor<N1, N2, N3, K1, K2a, K2b, T>";


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
contract3_eval_functor<N1, N2, N3, K1, K2a, K2b, T>::contract3_eval_functor(
    contract3_core<N1, N2, N3, K1, K2a, K2b, T> &core,
    const contract_subexpr_labels<N, M, K, T> &labels_ab,
    const letter_expr<ND> &label_d) :

    m_core(core),
    m_label_a(labels_ab.get_label_a()),
    m_label_b(labels_ab.get_label_b()),
    m_label_c(label_c),
    m_interm_a(core.get_expr_1(), m_label_a),
    m_interm_b(core.get_expr_2(), m_label_b),
    m_op(0), m_arg(0) {

}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
contract3_eval_functor<N1, N2, N3, K1, K2a, K2b, T>::~contract3_eval_functor() {

    delete m_op;
    delete m_arg;
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
void contract3_eval_functor<N1, N2, N3, K1, K2a, K2b, T>::evaluate() {

    m_interm_a.evaluate();
    m_interm_b.evaluate();
    m_interm_c.evaluate();

    arg<NA, T, tensor_tag> arga = m_interm_a.get_arg();
    arg<NB, T, tensor_tag> argb = m_interm_b.get_arg();
    arg<NC, T, tensor_tag> argc = m_interm_c.get_arg();

    contract_contraction2_builder<N, M, K> cb(
        m_label_a, permutation<NA>(arga.get_perm(), true),
        m_label_b, permutation<NB>(argb.get_perm(), true),
        m_label_c, m_core.get_contr());

    m_op = new btod_contract2<N, M, K>(cb.get_contr(),
        arga.get_btensor(), argb.get_btensor());
    m_arg = new arg<NC, T, oper_tag>(*m_op,
        arga.get_coeff() * argb.get_coeff());
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2a, size_t K2b,
    typename T>
void contract3_eval_functor<N1, N2, N3, K1, K2a, K2b, T>::clean() {

    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
    m_interm_a.clean();
    m_interm_b.clean();
}


} // namespace labeled_btensor_expr
} // namespace libtensor
#endif
#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT3_EVAL_FUNCTOR_H
