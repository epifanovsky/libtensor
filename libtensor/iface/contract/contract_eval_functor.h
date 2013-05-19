#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_H

#include <libtensor/block_tensor/btod_contract2.h>
#include "../expr/anon_eval.h"
#include "../expr/direct_eval.h"
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
    contract_eval_functor_base<N, M, K, T> *m_func;

public:
    contract_eval_functor(
        contract_core<N, M, K, T> &core,
        const contract_subexpr_labels<N, M, K, T> &labels_ab,
        const letter_expr<NC> &label_c);

    ~contract_eval_functor();

    void evaluate() {
        m_func->evaluate();
    }

    void clean() {
        m_func->clean();
    }

    arg<NC, T, oper_tag> get_arg() const {
        return m_func->get_arg();
    }

};


} // namespace labeled_btensor_expr
} // namespace libtensor

#include "contract_eval_functor_xxxx.h"
#include "contract_eval_functor_xx10.h"
#include "contract_eval_functor_10xx.h"
#include "contract_eval_functor_1010.h"

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, size_t K, typename T>
const char contract_eval_functor<N, M, K, T>::k_clazz[] =
    "contract_eval_functor<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
contract_eval_functor<N, M, K, T>::contract_eval_functor(
    contract_core<N, M, K, T> &core,
    const contract_subexpr_labels<N, M, K, T> &labels_ab,
    const letter_expr<NC> &label_c) {

    std::auto_ptr< eval_container_i<NA, T> > conta(core.get_expr_1().
        get_core().create_container(labels_ab.get_label_a()));
    std::auto_ptr< eval_container_i<NB, T> > contb(core.get_expr_2().
        get_core().create_container(labels_ab.get_label_b()));

    bool ta = (conta->get_ntensor() == 1 && conta->get_noper() == 0);
    bool tb = (contb->get_ntensor() == 1 && contb->get_noper() == 0);
    
    if(ta && tb) {
        m_func = new contract_eval_functor_1010<N, M, K, T>(
            core, labels_ab, label_c);
    } else if(ta) {
        m_func = new contract_eval_functor_10xx<N, M, K, T>(
            core, labels_ab, label_c);
    } else if(tb) {
        m_func = new contract_eval_functor_xx10<N, M, K, T>(
            core, labels_ab, label_c);
    } else {
        m_func = new contract_eval_functor_xxxx<N, M, K, T>(
            core, labels_ab, label_c);
    }
}


template<size_t N, size_t M, size_t K, typename T>
contract_eval_functor<N, M, K, T>::~contract_eval_functor() {

    delete m_func;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_EVAL_FUNCTOR_H
