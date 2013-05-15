#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_CORE_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, typename T> class direct_product_eval;


/** \brief Direct product operation expression core
    \tparam N Order of the first tensor (A).
    \tparam M Order of the second tensor (B).

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class direct_product_core : public expr_core_i<N + M, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
     //! Evaluating container type
    typedef direct_product_eval<N, M, T> eval_container_t;

private:
    expr<N, T> m_expr1; //!< First expression
    expr<M, T> m_expr2; //!< Second expression
    sequence<N + M, const letter*> m_defout; //!< Default output label

public:
    /** \brief Creates the expression core
        \param expr1 First expression (A).
        \param expr2 Second expression (B).
        \throw expr_exception If letters are inconsistent.
     **/
    direct_product_core(const expr<N, T> &expr1, const expr<M, T> &expr2);

    /** \brief Returns the first expression (A)
     **/
    expr<N, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (A), const version
     **/
    const expr<N, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression (B)
     **/
    expr<M, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (B), const version
     **/
    const expr<M, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Returns whether the result's label contains a letter
        \param let Letter.
     **/
    bool contains(const letter &let) const;

    /** \brief Returns the index of a letter in the result's label
        \param let Letter.
        \throw expr_exception If the label does not contain the
            requested letter.
     **/
    size_t index_of(const letter &let) const;

    /** \brief Returns the letter at a given position in the result's label
        \param i Letter index.
        \throw out_of_bounds If the index is out of bounds.
     **/
    const letter &letter_at(size_t i) const;

};


/** \brief Evaluating container for the direct product of two tensors
    \tparam N Order of the first %tensor (A).
    \tparam M Order of the second %tensor (B).
    \tparam Expr1 First expression (A) type.
    \tparam Expr2 Second expression (B) type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, typename E1, typename E2>
class direct_product_eval : public eval_i<N + M, T> {
public:
    static const char *k_clazz; //!< Class name
    static const size_t k_ordera = N; //!< Order of the first %tensor
    static const size_t k_orderb = M; //!< Order of the second %tensor
    static const size_t k_orderc = N + M; //!< Order of the result

    //!    Contraction expression core type
    typedef direct_product_core<N, M, T, E1, E2> core_t;

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
    typedef direct_product_subexpr_labels<N, M, T, E1, E2> subexpr_labels_t;

    //!    Evaluating functor type (specialized for A and B)
    typedef direct_product_eval_functor<N, M, T, E1, E2,
        k_narg_tensor_a, k_narg_oper_a, k_narg_tensor_b, k_narg_oper_b>
        functor_t;

    //!    Number of arguments in the expression
    template<typename Tag, int Dummy = 0>
    struct narg {
        static const size_t k_narg = 0;
    };

private:
    subexpr_labels_t m_sub_labels;
    functor_t m_func; //!< Sub-expression evaluation functor

public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    direct_product_eval(
        expression_t &expr, const letter_expr<k_orderc> &label)
        throw(exception);

    /** \brief Virtual destructor
     **/
    virtual ~direct_product_eval() { }

    /** \brief Evaluates sub-expressions into temporary tensors
     **/
    void prepare();

    /** \brief Cleans temporary tensors
     **/
    void clean();

    template<typename Tag>
    arg<N + M, T, Tag> get_arg(const Tag &tag, size_t i) const
        throw(exception);

    /** \brief Returns a single argument
     **/
    arg<N + M, T, oper_tag> get_arg(const oper_tag &tag, size_t i) const
        throw(exception);

private:
    static contraction2<N, M, 0> mk_contr(expression_t &expr,
        labeled_btensor<k_orderc, T, true> &result)
        throw(exception);
};


template<size_t N, size_t M, typename T>
const char direct_product_core<N, M, T>::k_clazz[] =
    "direct_product_core<N, M, T>";


template<size_t N, size_t M, typename T>
direct_product_core<N, M, T>::direct_product_core(
    const expr<N, T> &expr1, const expr<M, T> &expr2) :

    m_expr1(expr1), m_expr2(expr2), m_defout(0) {

    static const char method[] =
        "direct_product_core(const expr<N, T>&, const expr<M, T>&)";

    for(size_t i = 0; i < N; i++) {
        const letter &l = expr1.letter_at(i);
        if(expr2.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Duplicate letter index.");
        } else {
            m_defout[i] = &l;
        }
    }
    for(size_t i = 0; i < M; i++) {
        const letter &l = expr2.letter_at(i);
        if(expr1.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Duplicate letter index.");
        } else {
            m_defout[i + N] = &l;
        }
    }
}


template<size_t N, size_t M, typename T>
bool direct_product_core<N, M, T>::contains(const letter &let) const {

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N, size_t M, typename T>
size_t direct_product_core<N, M, T>::index_of(const letter &let) const {

    static const char method[] = "index_of(const letter&)";

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Expression doesn't contain the letter.");
}


template<size_t N, size_t M, typename T>
const letter&direct_product_core<N, M, T>::letter_at(size_t i) const {

    static const char method[] = "letter_at(size_t)";

    if(i >= N + M) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


template<size_t N, size_t M, typename T, typename E1, typename E2>
const char *direct_product_eval<N, M, T, E1, E2>::k_clazz =
    "direct_product_eval<N, M, T, E1, E2>";


template<size_t N, size_t M, typename T, typename E1, typename E2>
template<int Dummy>
struct direct_product_eval<N, M, T, E1, E2>::narg<oper_tag, Dummy> {
    static const size_t k_narg = 1;
};


template<size_t N, size_t M, typename T, typename E1, typename E2>
inline direct_product_eval<N, M, T, E1, E2>::direct_product_eval(
    expression_t &expr, const letter_expr<k_orderc> &label)
    throw(exception) :

    m_sub_labels(expr, label),
    m_func(expr, m_sub_labels, label) {

}


template<size_t N, size_t M, typename T, typename E1, typename E2>
inline void direct_product_eval<N, M, T, E1, E2>::prepare() {

    m_func.evaluate();
}


template<size_t N, size_t M, typename T, typename E1, typename E2>
inline void direct_product_eval<N, M, T, E1, E2>::clean() {

    m_func.clean();
}


template<size_t N, size_t M, typename T, typename E1, typename E2>
template<typename Tag>
arg<N + M, T, Tag> direct_product_eval<N, M, T, E1, E2>::get_arg(
    const Tag &tag, size_t i) const throw(exception) {

    static const char *method = "get_arg(const Tag&, size_t)";
    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, size_t M, typename T, typename E1, typename E2>
arg<N + M, T, oper_tag> direct_product_eval<N, M, T, E1, E2>::get_arg(
    const oper_tag &tag, size_t i) const throw(exception) {

    static const char *method = "get_arg(const oper_tag&, size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

    return m_func.get_arg();
}


template<size_t N, size_t M, typename T, typename E1, typename E2>
contraction2<N, M, 0> direct_product_eval<N, M, T, E1, E2>::mk_contr(
    expression_t &expr, labeled_btensor<k_orderc, T, true> &result)
    throw(exception) {

    size_t seq1[N + M], seq2[N + M];
    for(size_t i = 0; i < N + M; i++) {
        seq1[i] = i;
        seq2[i] = expr.index_of(result.letter_at(i));
    }
    permutation_builder<N + M> permc(seq1, seq2);
    contraction2<N, M, 0> contr(permc.get_perm());
    return contr;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_CORE_H
