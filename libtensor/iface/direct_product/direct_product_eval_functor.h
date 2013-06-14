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
//    anon_eval_a_t m_eval_a; //!< Anonymous evaluator for sub-expression A
//    permutation<k_ordera> m_invperm_a;
//    anon_eval_b_t m_eval_b; //!< Anonymous evaluator for sub-expression B
//    permutation<k_orderb> m_invperm_b;
//    direct_product_contraction2_builder<N, M> m_contr_bld; //!< Contraction builder
//    btod_contract2<N, M, 0> *m_op; //!< Contraction operation
//    arg<k_orderc, T, oper_tag> *m_arg; //!< Composed operation argument

public:
    direct_product_eval_functor(
        direct_product_core<N, M, T> &core,
        const direct_product_subexpr_labels<N, M, T> &labels_ab,
        const letter_expr<NC> &label_c);

    ~direct_product_eval_functor();

    void evaluate();

    void clean();

    arg<NC, T, oper_tag> get_arg() const { return *m_arg; }

private:
    void create_arg();
    void destroy_arg();

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

//    m_eval_a(expr.get_core().get_expr_1(), labels_ab.get_label_a()),
//    m_eval_b(expr.get_core().get_expr_2(), labels_ab.get_label_b()),
//    m_contr_bld(labels_ab.get_label_a(), m_invperm_a,
//        labels_ab.get_label_b(), m_invperm_b, label_c),
//    m_op(0), m_arg(0) {

}


template<size_t N, size_t M, typename T>
direct_product_eval_functor<N, M, T>::~direct_product_eval_functor() {

    delete m_arg;
    delete m_op;
}


template<size_t N, size_t M, typename T>
void direct_product_eval_functor<N, M, T>::evaluate() {

//    m_eval_a.evaluate();
//    m_eval_b.evaluate();
//    create_arg();

    m_interm_a.evaluate();
    m_interm_b.evaluate();

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


//template<size_t N, size_t M, typename T, typename E1, typename E2,
//size_t NT1, size_t NO1, size_t NT2, size_t NO2>
//void direct_product_eval_functor<N, M, T, E1, E2, NT1, NO1, NT2, NO2>::clean() {
//
//    destroy_arg();
//    m_eval_a.clean();
//    m_eval_b.clean();
//}
//
//
//template<size_t N, size_t M, typename T, typename E1, typename E2,
//size_t NT1, size_t NO1, size_t NT2, size_t NO2>
//void direct_product_eval_functor<N, M, T, E1, E2, NT1, NO1, NT2, NO2>::create_arg() {
//
//    destroy_arg();
//    m_op = new btod_contract2<N, M, 0>(m_contr_bld.get_contr(),
//        m_eval_a.get_btensor(), m_eval_b.get_btensor());
//    m_arg = new arg<k_orderc, T, oper_tag>(*m_op, 1.0);
//}
//
//
//template<size_t N, size_t M, typename T, typename E1, typename E2,
//size_t NT1, size_t NO1, size_t NT2, size_t NO2>
//void direct_product_eval_functor<N, M, T, E1, E2, NT1, NO1, NT2, NO2>::destroy_arg() {
//
//    delete m_arg; m_arg = 0;
//    delete m_op; m_op = 0;
//}
//

} // namespace labeled_btensor_expr
} // namespace libtensor

// Template specializations
//#include "direct_product_eval_functor_xx10.h"
//#include "direct_product_eval_functor_10xx.h"
//#include "direct_product_eval_functor_1010.h"

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_EVAL_FUNCTOR_H
