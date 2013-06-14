#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_H

#include "diag_subexpr_label_builder.h"
#include "diag_params_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Functor for evaluating the diagonal

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class diag_eval_functor_base {
public:
    virtual ~diag_eval_functor_base() { }

    virtual void evaluate() = 0;

    virtual void clean() = 0;

    virtual arg<N - M + 1, T, oper_tag> get_arg() const = 0;

};


/** \brief Functor for evaluating the diagonal

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class diag_eval_functor {
public:
    static const char k_clazz[]; //!< Class name

private:
    diag_eval_functor_base<N, M, T> *m_func;

public:
    diag_eval_functor(
        const diag_core<N, M, T> &core,
        const diag_subexpr_label_builder<N, M> &labels_a,
        const letter_expr<N - M + 1> &label_b);

    ~diag_eval_functor();

    void evaluate() {
        m_func->evaluate();
    }

    void clean() {
        m_func->clean();
    }

    arg<N - M + 1, T, oper_tag> get_arg() const {
        return m_func->get_arg();
    }

};


} // namespace labeled_btensor_expr
} // namespace libtensor

#include "diag_eval_functor_xx.h"
#include "diag_eval_functor_10.h"

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, typename T>
const char diag_eval_functor<N, M, T>::k_clazz[] =
    "diag_eval_functor<N, M, T>";


template<size_t N, size_t M, typename T>
diag_eval_functor<N, M, T>::diag_eval_functor(
    const diag_core<N, M, T> &core,
    const diag_subexpr_label_builder<N, M> &labels_a,
    const letter_expr<N - M + 1> &label_b) {

    std::auto_ptr< eval_container_i<N, T> > conta(
        core.get_sub_expr().get_core().create_container(labels_a.get_label()));

    bool ta = (conta->get_ntensor() == 1 && conta->get_noper() == 0);

    if(ta) {
        m_func = new diag_eval_functor_10<N, M, T>(core, labels_a, label_b);
    } else {
        m_func = new diag_eval_functor_xx<N, M, T>(core, labels_a, label_b);
    }
}


template<size_t N, size_t M, typename T>
diag_eval_functor<N, M, T>::~diag_eval_functor() {

    delete m_func;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_FUNCTOR_H
