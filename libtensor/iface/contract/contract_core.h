#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_CORE_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Contraction operation expression core
    \tparam N Order of the first tensor (A) less contraction degree.
    \tparam M Order of the second tensor (B) less contraction degree.
    \tparam K Number of indexes contracted.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class contract_core : public expr_core_i<N + M, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
     //! Evaluating container type
    typedef contract_eval<N, M, K, T> eval_container_t;

private:
    letter_expr<K> m_contr; //!< Contracted indexes
    expr<N + K, T> m_expr1; //!< First expression
    expr<M + K, T> m_expr2; //!< Second expression
    sequence<N + M, const letter*> m_defout; //!< Default output label

public:
    /** \brief Creates the expression core
        \param contr Letter expression indicating which indexes will be
            contracted.
        \param expr1 First expression (A).
        \param expr2 Second expression (B).
        \throw expr_exception If letters are inconsistent.
     **/
    contract_core(const letter_expr<K> &contr,
        const expr<N + K, T> &expr1, const expr<M + K, T> &expr2);

    /** \brief Virtual destructor
     **/
    virtual ~contract_core() { }

    /** \brief Clones this object using new
     **/
    expr_core_i<N + M, T> *clone() const {
        return new contract_core(*this);
    }

    /** \brief Returns the first expression (A)
     **/
    E1 &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (A), const version
     **/
    const E1 &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression (B)
     **/
    E2 &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (B), const version
     **/
    const E2 &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Returns the contracted indexes
     **/
    const letter_expr<K> &get_contr() const {
        return m_contr;
    }

    /** \brief Returns whether the result's label contains a %letter
        \param let Letter.
     **/
    bool contains(const letter &let) const;

    /** \brief Returns the %index of a %letter in the result's label
        \param let Letter.
        \throw expr_exception If the label does not contain the
            requested letter.
     **/
    size_t index_of(const letter &let) const throw(expr_exception);

    /** \brief Returns the %letter at a given position in
            the result's label
        \param i Letter index.
        \throw out_of_bounds If the index is out of bounds.
     **/
    const letter &letter_at(size_t i) const throw(out_of_bounds);

};


/** \brief Evaluating container for the contraction of two tensors
    \tparam N Order of the first tensor (A) less contraction degree.
    \tparam M Order of the second tensor (B) less contraction degree.
    \tparam K Number of indexes contracted.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class contract_eval : public eval_i<N + M, T> {
public:
    static const char k_clazz[]; //!< Class name
    static const size_t k_ordera = N + K; //!< Order of the first %tensor
    static const size_t k_orderb = M + K; //!< Order of the second %tensor
    static const size_t k_orderc = N + M; //!< Order of the result

    enum {
        NC = N + M
    };

    //!    Contraction expression core type
    typedef core_contract<N, M, K, T, E1, E2> core_t;

    //!    Contraction expression type
    typedef expr<k_orderc, T, core_t> expression_t;

    //!    Evaluating container type of the first expression (A)
    typedef typename E1::eval_container_t eval_container_a_t;

    //!    Evaluating container type of the second expression (B)
    typedef typename E2::eval_container_t eval_container_b_t;

    //!    Number of %tensor arguments in expression A
    static const size_t k_narg_tensor_a =
        eval_container_a_t::template narg<tensor_tag>::k_narg;

    //!    Number of operation arguments in expression A
    static const size_t k_narg_oper_a =
        eval_container_a_t::template narg<oper_tag>::k_narg;

    //!    Number of %tensor arguments in expression B
    static const size_t k_narg_tensor_b =
        eval_container_b_t::template narg<tensor_tag>::k_narg;

    //!    Number of operation arguments in expression A
    static const size_t k_narg_oper_b =
        eval_container_b_t::template narg<oper_tag>::k_narg;

    //!    Labels for sub-expressions
    typedef contract_subexpr_labels<N, M, K, T, E1, E2> subexpr_labels_t;

    //!    Evaluating functor type (specialized for A and B)
    typedef contract_eval_functor<N, M, K, T, E1, E2,
        k_narg_tensor_a, k_narg_oper_a, k_narg_tensor_b, k_narg_oper_b>
        functor_t;

    //!    Number of arguments in the expression
    template<typename Tag, int Dummy = 0>
    struct narg {
        static const size_t k_narg = 0;
    };

private:
    expr<NC, T> m_expr; //!< Scaled expression
    contract_core<N, M, K, T> &m_core; //!< Expression core
    subexpr_labels_t m_sub_labels;
    functor_t m_func; //!< Sub-expression evaluation functor


public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    contract_eval(expr<NC, T> &e, const letter_expr<NC> &label);

    /** \brief Virtual destructor
     **/
    virtual ~contract_eval() { }

    /** \brief Evaluates sub-expressions into temporary tensors
     **/
    virtual void prepare();

    /** \brief Cleans up temporary tensors
     **/
    virtual void clean();

    template<typename Tag>
    arg<N + M, T, Tag> get_arg(const Tag &tag, size_t i) const
        throw(exception);

    /** \brief Returns a single argument
     **/
    arg<N + M, T, oper_tag> get_arg(const oper_tag &tag, size_t i) const
        throw(exception);

};


template<size_t N, size_t M, size_t K, typename T>
const char contract_core<N, M, K, T>::k_clazz[] = "contract_core<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
contract_core<N, M, K, T>::contract_core(
    const letter_expr<K> &contr,
    const expr<N + K, T> &expr1,
    const expr<M + K, T> &expr2) :

    m_contr(contr), m_expr1(expr1), m_expr2(expr2), m_defout(0) {

    static const char method[] = "contract_core(const letter_expr<K>&, "
        "const expr<N + K, T>&, const expr<M + K, T>&)";

    for(size_t i = 0; i < K; i++) {
        const letter &l = contr.letter_at(i);
        if(!expr1.contains(l) || !expr2.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Contracted index is absent from arguments.");
        }
    }

    size_t j = 0;
    for(size_t i = 0; i < N + K; i++) {
        const letter &l = expr1.letter_at(i);
        if(!contr.contains(l)) {
            if(expr2.contains(l)) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Duplicate uncontracted index in A.");
            } else {
                m_defout[j++] = &l;
            }
        }
    }
    for(size_t i = 0; i < M + K; i++) {
        const letter &l = expr2.letter_at(i);
        if(!contr.contains(l)) {
            if(expr1.contains(l)) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Duplicate uncontracted index in B.");
            } else {
                m_defout[j++] = &l;
            }
        }
    }
}


template<size_t N, size_t M, size_t K, typename T>
bool contract_core<N, M, K, T>::contains(const letter &let) const {

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N, size_t M, size_t K, typename T>
size_t contract_core<N, M, K, T>::index_of(const letter &let) const {

    static const char method[] = "index_of(const letter&)";

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Expression doesn't contain the letter.");
}


template<size_t N, size_t M, size_t K, typename T>
const letter&contract_core<N, M, K, T>::letter_at(size_t i) const {

    static const char method[] = "letter_at(size_t)";

    if(i >= N + M) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


template<size_t N, size_t M, size_t K, typename T>
const char contract_eval<N, M, K, T>::k_clazz[] = "contract_eval<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
template<int Dummy>
struct contract_eval<N, M, K, T, E1, E2>::narg<oper_tag, Dummy> {
    static const size_t k_narg = 1;
};


template<size_t N, size_t M, size_t K, typename T>
contract_eval<N, M, K, T>::contract_eval(expr<NC, T> &e,
    const letter_expr<NC> &label) :

    m_expr(e),
    m_core(dynamic_cast< contract_core<N, M, K, T>& >(m_expr.get_core())),
    m_sub_labels(expr, label),
    m_func(expr, m_sub_labels, label) {

}


template<size_t N, size_t M, size_t K, typename T>
void contract_eval<N, M, K, T>::prepare() {

    m_func.evaluate();
}


template<size_t N, size_t M, size_t K, typename T>
void contract_eval<N, M, K, T>::clean() {

    m_func.clean();
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
template<typename Tag>
arg<N + M, T, Tag> contract_eval<N, M, K, T, E1, E2>::get_arg(
    const Tag &tag, size_t i) const throw(exception) {

    static const char *method = "get_arg(const Tag&, size_t)";
    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
arg<N + M, T, oper_tag> contract_eval<N, M, K, T, E1, E2>::get_arg(
    const oper_tag &tag, size_t i) const throw(exception) {

    static const char *method = "get_arg(const oper_tag&, size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

    return m_func.get_arg();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_CORE_H
