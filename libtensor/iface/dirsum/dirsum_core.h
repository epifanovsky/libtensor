#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_CORE_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Direct sum operation expression core
    \tparam N Order of the first tensor (A).
    \tparam M Order of the second tensor (B).

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class dirsum_core : public expr_core_i<N + M, T> {
public:
    static const char k_clazz[]; //!< Class name

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
    dirsum_core(const expr<N, T> &expr1, const expr<M, T> &expr2);

    /** \brief Virtual destructor
     **/
    virtual ~dirsum_core() { }

    /** \brief Clones this object using new
     **/
    virtual expr_core_i<N + M, T> *clone() const {
        return new dirsum_core<N, M, T>(*this);
    }

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

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N + M, T> *create_container(
        const letter_expr<N + M> &label) const;

    /** \brief Returns whether the result's label contains a letter
        \param let Letter.
     **/
    virtual bool contains(const letter &let) const;

    /** \brief Returns the index of a letter in the result's label
        \param let Letter.
        \throw expr_exception If the label does not contain the
            requested letter.
     **/
    virtual size_t index_of(const letter &let) const;

    /** \brief Returns the letter at a given position in the result's label
        \param i Letter index.
        \throw out_of_bounds If the index is out of bounds.
     **/
    virtual const letter &letter_at(size_t i) const;

};


} // namespace labeled_btensor_expr
} // namespace libtensor

#include "dirsum_subexpr_labels.h"
#include "dirsum_eval_functor.h"

namespace libtensor {
namespace labeled_btensor_expr {

/** \brief Evaluating container for the direct sum of two tensors
    \tparam N Order of the first tensor (A).
    \tparam M Order of the second tensor (B).

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class dirsum_eval : public eval_container_i<N + M, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NC = N + M
    };

private:
    dirsum_core<N, M, T> m_core; //!< Expression core
    dirsum_subexpr_labels<N, M, T> m_sub_labels;
    dirsum_eval_functor<N, M, T> m_func; //!< Sub-expression evaluation functor

public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    dirsum_eval(
        const dirsum_core<N, M, T> &core,
        const letter_expr<NC> &label);

    /** \brief Virtual destructor
     **/
    virtual ~dirsum_eval() { }

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


    /** \brief Returns tensor arguments (not valid)
        \param i Argument number.
     **/
    virtual arg<N + M, T, tensor_tag> get_tensor_arg(size_t i);

    /** \brief Returns operation arguments
        \param i Argument number (0 is the only valid value).
     **/
    virtual arg<N + M, T, oper_tag> get_oper_arg(size_t i);
};


template<size_t N, size_t M, typename T>
const char dirsum_core<N, M, T>::k_clazz[] = "dirsum_core<N, M, T>";


template<size_t N, size_t M, typename T>
dirsum_core<N, M, T>::dirsum_core(
    const expr<N, T> &expr1, const expr<M, T> &expr2) :

    m_expr1(expr1), m_expr2(expr2), m_defout(0) {

    static const char method[] =
        "dirsum_core(const expr<N, T>&, const expr<M, T>&)";

    const expr_core_i<N, T> &core1 = expr1.get_core();
    const expr_core_i<M, T> &core2 = expr2.get_core();

    size_t j = 0;
    for(size_t i = 0; i < N; i++) {
        const letter &l = core1.letter_at(i);
        if(core2.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Duplicate uncontracted index in A.");
        } else {
            m_defout[j++] = &l;
        }
    }
    for(size_t i = 0; i < M; i++) {
        const letter &l = core2.letter_at(i);
        if(core1.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Duplicate uncontracted index in B.");
        } else {
            m_defout[j++] = &l;
        }
    }
}


template<size_t N, size_t M, typename T>
eval_container_i<N + M, T> *dirsum_core<N, M, T>::create_container(
    const letter_expr<N + M> &label) const {

    return new dirsum_eval<N, M, T>(*this, label);
}


template<size_t N, size_t M, typename T>
bool dirsum_core<N, M, T>::contains(const letter &let) const {

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N, size_t M, typename T>
size_t dirsum_core<N, M, T>::index_of(const letter &let) const {

    static const char method[] = "index_of(const letter&)";

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Expression doesn't contain the letter.");
}


template<size_t N, size_t M, typename T>
const letter &dirsum_core<N, M, T>::letter_at(size_t i) const {

    static const char method[] = "letter_at(size_t)";

    if(i >= N + M) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


template<size_t N, size_t M, typename T>
const char dirsum_eval<N, M, T>::k_clazz[] = "dirsum_eval<N, M, T>";


template<size_t N, size_t M, typename T>
dirsum_eval<N, M, T>::dirsum_eval(
const dirsum_core<N, M, T> &core,
const letter_expr<NC> &label) :

m_core(core),
m_sub_labels(core, label),
m_func(m_core, m_sub_labels, label) {

}


template<size_t N, size_t M, typename T>
inline void dirsum_eval<N, M, T>::prepare() {

    m_func.evaluate();
}


template<size_t N, size_t M, typename T>
inline void dirsum_eval<N, M, T>::clean() {

    m_func.clean();
}


template<size_t N, size_t M, typename T>
arg<N + M, T, tensor_tag> dirsum_eval<N, M, T>::get_tensor_arg(size_t i) {

    static const char *method = "get_arg(size_t)";
    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, size_t M, typename T>
arg<N + M, T, oper_tag> dirsum_eval<N, M, T>::get_oper_arg(size_t i) {

    static const char *method = "get_arg(const oper_tag&, size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

    return m_func.get_arg();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_CORE_H
