#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_EVAL_FUNCTOR_H

#include <libtensor/block_tensor/btod_dirsum.h>
#include "../expr/anon_eval.h"
#include "dirsum_permutation_builder.h"
#include "dirsum_subexpr_labels.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating direct sums (base class)

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class dirsum_eval_functor_base {
public:
    enum {
        NC = N + M
    };

public:
    virtual ~dirsum_eval_functor_base() { }
    virtual void evaluate() = 0;
    virtual void clean() = 0;
    virtual arg<NC, T, oper_tag> get_arg() const = 0;

};

/** \brief Functor for evaluating direct sums

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class dirsum_eval_functor {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NC = N + M
    };

private:
    dirsum_core<N, M, T> m_core; //!< Direct sum core
    letter_expr<N> m_label_a;
    letter_expr<M> m_label_b;
    letter_expr<N + M> m_label_c;
    interm<N, T> m_interm_a;
    interm<M, T> m_interm_b;

    btod_dirsum<N, M> *m_op; //!< Direct sum operation
    arg<NC, T, oper_tag> *m_arg; //!< Composed operation argument

public:
    dirsum_eval_functor(
        dirsum_core<N, M, T> &core,
        const dirsum_subexpr_labels<N, M, T> &labels_ab,
        const letter_expr<N + M> &label_c);

    ~dirsum_eval_functor();

    void evaluate();

    void clean();

    arg<NC, T, oper_tag> get_arg() const { return *m_arg; }
};


template<size_t N, size_t M, typename T>
const char dirsum_eval_functor<N, M, T>::k_clazz[] =
        "dirsum_eval_functor<N, M, T>";


template<size_t N, size_t M, typename T>
dirsum_eval_functor<N, M, T>::dirsum_eval_functor(
    dirsum_core<N, M, T> &core,
    const dirsum_subexpr_labels<N, M, T> &labels_ab,
    const letter_expr<N + M> &label_c) :

    m_core(core),
    m_label_a(labels_ab.get_label_a()),
    m_label_b(labels_ab.get_label_b()),
    m_label_c(label_c),
    m_interm_a(core.get_expr_1(), m_label_a),
    m_interm_b(core.get_expr_2(), m_label_b),
    m_op(0), m_arg(0) {

}


template<size_t N, size_t M, typename T>
dirsum_eval_functor<N, M, T>::~dirsum_eval_functor() {

    delete m_op;
    delete m_arg;
}


template<size_t N, size_t M, typename T>
void dirsum_eval_functor<N, M, T>::evaluate() {

    m_interm_a.evaluate();
    m_interm_b.evaluate();

    arg<N, T, tensor_tag> arga = m_interm_a.get_arg();
    arg<M, T, tensor_tag> argb = m_interm_b.get_arg();

    dirsum_permutation_builder<N, M> pb(
        m_label_a, permutation<N>(arga.get_perm(), true),
        m_label_b, permutation<M>(argb.get_perm(), true),
        m_label_c);

    m_op = new btod_dirsum<N, M>(
            arga.get_btensor(), arga.get_coeff(),
            argb.get_btensor(), argb.get_coeff(),
            pb.get_perm());
    m_arg = new arg<NC, T, oper_tag>(*m_op, 1.0);
}


template<size_t N, size_t M, typename T>
void dirsum_eval_functor<N, M, T>::clean() {

    delete m_op; m_op = 0;
    delete m_arg; m_arg = 0;
}




} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_EVAL_FUNCTOR_H
