#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_EVAL_FUNCTOR_H

#include <libtensor/block_tensor/btod_ewmult2.h>
#include "../expr/anon_eval.h"
#include "../expr/direct_eval.h"
#include "ewmult_core.h"
#include "ewmult_perm_builder.h"
#include "ewmult_subexpr_labels.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating element-wise products

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class ewmult_eval_functor {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K,
        NB = M + K,
        NC = N + M + K
    };

private:
    ewmult_core<N, M, K, T> m_core;
    letter_expr<NA> m_label_a;
    letter_expr<NB> m_label_b;
    letter_expr<NC> m_label_c;
    interm<NA, T> m_interm_a;
    interm<NB, T> m_interm_b;
    btod_ewmult2<N, M, K> *m_op; //!< Operation
    arg<NC, T, oper_tag> *m_arg; //!< Composed operation argument

public:
    ewmult_eval_functor(
        ewmult_core<N, M, K, T> &expr,
        const ewmult_subexpr_labels<N, M, K, T> &labels_ab,
        const letter_expr<NC> &label_c);

    ~ewmult_eval_functor();

    void evaluate();

    void clean();

    arg<N + M + K, T, oper_tag> get_arg() const { return *m_arg; }
};


template<size_t N, size_t M, size_t K, typename T>
const char ewmult_eval_functor<N, M, K, T>::k_clazz[] =
         "ewmult_eval_functor<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
ewmult_eval_functor<N, M, K, T>::ewmult_eval_functor(
    ewmult_core<N, M, K, T> &core,
    const ewmult_subexpr_labels<N, M, K, T> &labels_ab,
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
ewmult_eval_functor<N, M, K, T>::~ewmult_eval_functor() {

    delete m_op; m_op = 0;
    delete m_arg; m_arg = 0;
}


template<size_t N, size_t M, size_t K, typename T>
void ewmult_eval_functor<N, M, K, T>::evaluate() {

    m_interm_a.evaluate();
    m_interm_b.evaluate();

    arg<NA, T, tensor_tag> arga = m_interm_a.get_arg();
    arg<NB, T, tensor_tag> argb = m_interm_b.get_arg();

    ewmult_perm_builder<N, M, K> pb(
        m_label_a, permutation<NA>(arga.get_perm(), true),
        m_label_b, permutation<NB>(argb.get_perm(), true),
        m_label_c, m_core.get_ewidx());

    m_op = new btod_ewmult2<N, M, K>(
            arga.get_btensor(), pb.get_perma(),
            argb.get_btensor(), pb.get_permb(),
            pb.get_permc());
    m_arg = new arg<NC, T, oper_tag>(*m_op,
            arga.get_coeff() * argb.get_coeff());
}


template<size_t N, size_t M, size_t K, typename T>
void ewmult_eval_functor<N, M, K, T>::clean() {

    delete m_op; m_op = 0;
    delete m_arg; m_arg = 0;
    m_interm_a.clean();
    m_interm_b.clean();
}


} // namespace labeled_btensor_expr
} // namespace libtensor


#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_EVAL_FUNCTOR_H
