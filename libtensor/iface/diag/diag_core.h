#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_CORE_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Expression core for the extraction of a diagonal
    \tparam N Tensor order.
    \tparam M Diagonal order.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class diag_core : public expr_core_i<N - M + 1, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    const letter &m_diag_let; //!< Diagonal letter
    letter_expr<M> m_diag_lab; //!< Indexes defining a diagonal
    expr<N, T> m_subexpr; //!< Sub-expression
    sequence<N - M + 1, const letter*> m_defout; //!< Default output label

public:
    /** \brief Creates the expression core
        \param diag_letter Letter in the output.
        \param diag_label Expression defining the diagonal.
        \param subexpr Sub-expression.
     **/
    diag_core(const letter &diag_letter, const letter_expr<M> &diag_label,
        const expr<N, T> &subexpr);

    /** \brief Virtual destructor
     **/
    virtual ~diag_core() { }

    /** \brief Clones this object using new
     **/
    virtual expr_core_i<N - M + 1, T> *clone() const {
        return new diag_core(*this);
    }

    /** \brief Returns the diagonal letter
     **/
    const letter &get_diag_letter() const {
        return m_diag_let;
    }

    /** \brief Returns the diagonal indexes
     **/
    const letter_expr<M> &get_diag_label() const {
        return m_diag_lab;
    }

    /** \brief Returns the sub-expression
     **/
    expr<N, T> &get_sub_expr() {
        return m_subexpr;
    }

    /** \brief Returns the sub-expression, const version
     **/
    const expr<N, T> &get_sub_expr() const {
        return m_subexpr;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N - M + 1, T> *create_container(
        const letter_expr<N - M + 1> &label) const;

    /** \brief Returns whether the result's label contains a %letter
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


} // namespace labeled_btensor_expr
} // namespace libtensor

#include "diag_eval_functor.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Evaluating container for the extraction of a diagonal
    \tparam N Tensor order.
    \tparam M Diagonal order.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class diag_eval : public eval_i<N - M + 1, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    expr<N - M + 1, T> m_expr; //!< Expression
    diag_core<N, M, T> &m_core; //!< Expression core
    diag_subexpr_label_builder<N, M> m_sub_label; //!< Sub-expression label
    diag_eval_functor<N, M, T> m_func; //!< Specialized evaluation functor

public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    diag_eval(
        const expr<N - M + 1, T> &e,
        const letter_expr<N - M + 1> &label);

    /** \brief Virtual destructor
     **/
    virtual ~diag_eval() { }

    /** \brief Evaluates sub-expressions into temporary tensors
     **/
    virtual void prepare();

    /** \brief Cleans up temporary tensors
     **/
    virtual void clean();

    /** \brief Returns the number of tensors in expression
     **/
    virtual size_t get_ntensor() const {
        return 0;
    }

    /** \brief Returns the number of tensor operations in expression
     **/
    virtual size_t get_noper() const {
        return 1;
    }

    /** \brief Returns tensor arguments
        \param i Argument number.
     **/
    virtual arg<N - M + 1, T, tensor_tag> get_tensor_arg(size_t i) const;

    /** \brief Returns operation arguments
        \param i Argument number.
     **/
    virtual arg<N - M + 1, T, oper_tag> get_oper_arg(size_t i) const;

};


template<size_t N, size_t M, typename T>
const char diag_core<N, M, T>::k_clazz[] = "diag_core<N, M, T>";


template<size_t N, size_t M, typename T>
diag_core<N, M, T>::diag_core(const letter &diag_letter,
    const letter_expr<M> &diag_label, const expr<N, T> &subexpr) :

    m_diag_let(diag_letter), m_diag_lab(diag_label), m_subexpr(subexpr),
    m_defout(0) {

    static const char method[] =
        "diag_core(const letter&, const letter_expr<M>&, const expr<N, T>&)";

    for(size_t i = 0; i < M - 1; i++) {
        for(size_t j = i + 1; j < M; j++) {
            if(m_diag_lab.letter_at(i) == m_diag_lab.letter_at(j)) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Repetitive indexes.");
            }
        }
    }
    for(size_t i = 0; i < M; i++) {
        if(!m_expr.contains(m_diag_lab.letter_at(i))) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Bad index in diagonal.");
        }
    }

    size_t j = 0;
    bool first = true;
    for(size_t i = 0; i < N; i++) {
        const letter &l = m_expr.letter_at(i);
        bool indiag = m_diag_lab.contains(l);
        if(!indiag) m_defout[j++] = &l;
        else if(first && indiag) {
            m_defout[j++] = &m_diag_let;
            first = false;
        }
    }
}


template<size_t N, size_t M, typename T>
bool diag_core<N, M, T>::contains(const letter &let) const {

    for(register size_t i = 0; i < N - M + 1; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N, size_t M, typename T>
size_t diag_core<N, M, T>::index_of(const letter &let) const {

    static const char method[] = "index_of(const letter&)";

    for(register size_t i = 0; i < N - M + 1; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Bad letter.");
}


template<size_t N, size_t M, typename T>
const letter &diag_core<N, M, T>::letter_at(size_t i) const {

    static const char method[] = "letter_at(size_t)";

    if(i >= N - M + 1) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


template<size_t N, size_t M, typename T>
const char diag_eval<N, M, T>::k_clazz[] = "diag_eval<N, M, T>";


template<size_t N, size_t M, typename T>
diag_eval<N, M, T>::diag_eval(const expr<N - M + 1, T> &e,
    const letter_expr<N - M + 1> &label) :

    m_expr(e),
    m_core(dynamic_cast< diag_core<N, M, T>& >(m_expr.get_core())),
    m_sub_label(label, expr.get_core().get_diag_letter(),
        expr.get_core().get_diag_label()),
    m_func(expr, m_sub_label, label) {

}


template<size_t N, size_t M, typename T>
void diag_eval<N, M, T>::prepare() {

    m_func.evaluate();
}


template<size_t N, size_t M, typename T>
void diag_eval<N, M, T>::clean() {

    m_func.clean();
}


template<size_t N, size_t M, typename T>
arg<N - M + 1, T, tensor_tag> diag_eval<N, M, T>::get_tensor_arg(
    size_t i) const {

    static const char method[] = "get_tensor_arg(size_t)";

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, size_t M, typename T>
arg<N - M + 1, T, oper_tag> diag_eval<N, M, T>::get_oper_arg(
    size_t i) const {

    static const char method[] = "get_oper_arg(size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "i");
    }

    return m_func.get_arg();
}


template<size_t N, size_t M, typename T>
eval_container_i<N - M + 1, T> *diag_core<N, M, T>::create_container(
    const letter_expr<N - M + 1> &label) const {

    return new diag_eval<N, M, T>(*this, label);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_CORE_H
