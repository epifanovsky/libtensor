#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_EVAL_FUNCTOR_H

#include <libtensor/block_tensor/btod_contract2.h>
#include "../expr/interm.h"
#include "direct_product_subexpr_labels.h"
#include "direct_product_contraction2_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating direct products

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class direct_product_eval_functor {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N,
        NB = M,
        NC = N + M
    };

private:
    direct_product_core<N, M, T> &m_core;
    letter_expr<NA> m_label_a;
    letter_expr<NB> m_label_b;
    letter_expr<NC> m_label_c;
    interm<NA, T> m_interm_a;
    interm<NB, T> m_interm_b;
    btod_contract2<N, M, 0> *m_op; //!< Contraction operation
    arg<NC, T, oper_tag> *m_arg; //!< Composed operation argument

public:
    direct_product_eval_functor(
        direct_product_core<N, M, T> &core,
        const direct_product_subexpr_labels<N, M, T> &labels_ab,
        const letter_expr<NC> &label_c);

    ~direct_product_eval_functor();

    void evaluate();

    void clean();

    arg<N + M, T, oper_tag> get_arg() const { return *m_arg; }
};


template<size_t N, size_t M, typename T>
const char direct_product_eval_functor<N, M, T>::k_clazz[] =
    "direct_product_eval_functor<N, M, T>";


template<size_t N, size_t M, typename T>
direct_product_eval_functor<N, M, T>::direct_product_eval_functor(
    direct_product_core<N, M, T> &core,
    const direct_product_subexpr_labels<N, M, T> &labels_ab,
    const letter_expr<NC> &label_c) :

    m_core(core),
    m_label_a(labels_ab.get_label_a()),
    m_label_b(labels_ab.get_label_b()),
    m_label_c(label_c),
    m_interm_a(core.get_expr_1(), m_label_a),
    m_interm_b(core.get_expr_2(), m_label_b),
    m_op(0), m_arg(0) {

}


template<size_t N, size_t M, typename T>
direct_product_eval_functor<N, M, T>::~direct_product_eval_functor() {

    delete m_arg;
    delete m_op;
}


template<size_t N, size_t M, typename T>
void direct_product_eval_functor<N, M, T>::evaluate() {

    m_interm_a.evaluate();
    m_interm_b.evaluate();

    if (m_op != 0) delete m_op;
    if (m_arg != 0) delete m_arg;

    arg<NA, T, tensor_tag> arga = m_interm_a.get_arg();
    arg<NB, T, tensor_tag> argb = m_interm_b.get_arg();

    direct_product_contraction2_builder<N, M> cb(
        m_label_a, permutation<NA>(arga.get_perm(), true),
        m_label_b, permutation<NB>(argb.get_perm(), true),
        m_label_c);

    m_op = new btod_contract2<N, M, 0>(cb.get_contr(),
        arga.get_btensor(), argb.get_btensor());
    m_arg = new arg<NC, T, oper_tag>(*m_op,
        arga.get_coeff() * argb.get_coeff());
}


template<size_t N, size_t M, typename T>
void direct_product_eval_functor<N, M, T>::clean() {

    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
    m_interm_a.clean();
    m_interm_b.clean();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_EVAL_FUNCTOR_H
