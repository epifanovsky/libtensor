#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_CORE_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Element-wise product operation expression core
    \tparam N Order of the first tensor (A) less number of shared indexes.
    \tparam M Order of the second tensor (B) less number of shared indexes.
    \tparam K Number of shared indexes.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class ewmult_core : public expr_core_i<N + M + K, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K,
        NB = M + K,
        NC = N + M + K
    };

private:
    expr<NA, T> m_expr1; //!< First expression
    expr<NB, T> m_expr2; //!< Second expression
    letter_expr<K> m_ewidx; //!< Shared indexes
    sequence<NC, const letter*> m_defout; //!< Default output label

public:
    /** \brief Creates the expression core
        \param ewidx Letter expression indicating which indexes are shared.
        \param expr1 First expression (A).
        \param expr2 Second expression (B).
        \throw expr_exception If letters are inconsistent.
     **/
    ewmult_core(const letter_expr<K> &ewidx,
        const expr<NA, T> &expr1, const expr<NB, T> &expr2);

    /** \brief Virtual destructor
     **/
    virtual ~ewmult_core() { }

//    /** \brief Clones this object using new
//     **/
//    virtual expr_core_i<NC, T> *clone() const {
//        return new ewmult_core<N, M, K, T>(*this);
//    }

    /** \brief Returns the first expression (A)
     **/
    expr<NA, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (A), const version
     **/
    const expr<NA, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression (B)
     **/
    expr<NB, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (B), const version
     **/
    const expr<NB, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Returns the shared indexes
     **/
    const letter_expr<K> &get_ewidx() const {
        return m_ewidx;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N + M + K, T> *create_container(
        const letter_expr<N + M + K> &label) const;


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

#include "ewmult_subexpr_labels.h"
#include "ewmult_eval_functor.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Evaluating container for the element-wise product of two tensors
    \tparam N Order of the first tensor (A) less number of shared indexes.
    \tparam M Order of the second tensor (B) less number of shared indexes.
    \tparam K Number of shared indexes.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class ewmult_eval : public eval_container_i<N + M + K, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K,
        NB = M + K,
        NC = N + M + K
    };

private:
    ewmult_core<N, M, K, T> m_core; //!< Expression core
    ewmult_subexpr_labels<N, M, K, T> m_sub_labels;
    ewmult_eval_functor<N, M, K, T> m_func; //!< Sub-expression evaluation functor

public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    ewmult_eval(
        const ewmult_core<N, M, K, T> &core,
        const letter_expr<NC> &label);

    /** \brief Virtual destructor
     **/
    virtual ~ewmult_eval() { }

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
    virtual arg<N + M + K, T, tensor_tag> get_tensor_arg(size_t i);

    /** \brief Returns operation arguments
        \param i Argument number (0 is the only valid value).
     **/
    virtual arg<N + M + K, T, oper_tag> get_oper_arg(size_t i);
};


template<size_t N, size_t M, size_t K, typename T>
const char ewmult_core<N, M, K, T>::k_clazz[] = "ewmult_core<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
ewmult_core<N, M, K, T>::ewmult_core(const letter_expr<K> &ewidx,
    const expr<NA, T> &expr1, const expr<NB, T> &expr2) :

    m_expr1(expr1), m_expr2(expr2), m_ewidx(ewidx), m_defout(0) {

    static const char method[] = "ewmult_core(const letter_expr<K>&, "
        "const expr<N + K, T>&, const expr<M + K, T>&)";

    const expr_core_i<NA, T> &core1 = expr1.get_core();
    const expr_core_i<NB, T> &core2 = expr2.get_core();

    for(size_t i = 0; i < K; i++) {
        const letter &l = ewidx.letter_at(i);
        if(!core1.contains(l) || !core2.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Shared index is absent from arguments.");
        }
    }

    size_t j = 0;
    for(size_t i = 0; i < NA; i++) {
        const letter &l = core1.letter_at(i);
        if(!ewidx.contains(l)) {
            if(core2.contains(l)) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Duplicate index in A.");
            } else {
                m_defout[j++] = &l;
            }
        }
    }
    for(size_t i = 0; i < NB; i++) {
        const letter &l = core2.letter_at(i);
        if(!ewidx.contains(l)) {
            if(core1.contains(l)) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Duplicate index in B.");
            } else {
                m_defout[j++] = &l;
            }
        }
    }
    for(size_t i = 0; i < NA; i++) {
        const letter &l = core1.letter_at(i);
        if(ewidx.contains(l)) m_defout[j++] = &l;
    }
}


template<size_t N, size_t M, size_t K, typename T>
bool ewmult_core<N, M, K, T>::contains(const letter &let) const {

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N, size_t M, size_t K, typename T>
size_t ewmult_core<N, M, K, T>::index_of(const letter &let) const {

    static const char method[] = "index_of(const letter&)";

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Expression doesn't contain the letter.");
}


template<size_t N, size_t M, size_t K, typename T>
const letter &ewmult_core<N, M, K, T>::letter_at(size_t i) const {

    static const char method[] = "letter_at(size_t)";

    if(i >= N + M + K) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


template<size_t N, size_t M, size_t K, typename T>
const char ewmult_eval<N, M, K, T>::k_clazz[] = "ewmult_eval<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
ewmult_eval<N, M, K, T>::ewmult_eval(
    const ewmult_core<N, M, K, T> &core,
    const letter_expr<NC> &label) :

    m_core(core),
    m_sub_labels(core, label),
    m_func(m_core, m_sub_labels, label) {

}


template<size_t N, size_t M, size_t K, typename T>
void ewmult_eval<N, M, K, T>::prepare() {

    m_func.evaluate();
}


template<size_t N, size_t M, size_t K, typename T>
void ewmult_eval<N, M, K, T>::clean() {

    m_func.clean();
}


template<size_t N, size_t M, size_t K, typename T>
arg<N + M + K, T, tensor_tag> ewmult_eval<N, M, K, T>::get_tensor_arg(
    size_t i) {

    static const char *method = "get_tensor_arg(size_t)";
    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, size_t M, size_t K, typename T>
arg<N + M + K, T, oper_tag> ewmult_eval<N, M, K, T>::get_oper_arg(
    size_t i) {

    static const char *method = "get_oper_arg(size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

    return m_func.get_arg();
}


template<size_t N, size_t M, size_t K, typename T>
eval_container_i<N + M + K, T> *ewmult_core<N, M, K, T>::create_container(
    const letter_expr<N + M + K> &label) const {

    return new ewmult_eval<N, M, K, T>(*this, label);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_CORE_H
