#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_CORE_H

#include "../../defs.h"
#include "../../exception.h"
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
class ewmult_eval;


/** \brief Element-wise product operation expression core
    \tparam N Order of the first %tensor (A) less number of shared indexes.
    \tparam M Order of the second %tensor (B) less number of shared indexes.
    \tparam K Number of shared indexes.
    \tparam E1 First expression (A) type.
    \tparam E2 Second expression (B) type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
class ewmult_core {
public:
    static const char *k_clazz; //!< Class name

public:
     //!    Evaluating container type
    typedef ewmult_eval<N, M, K, T, E1, E2> eval_container_t;

private:
    E1 m_expr1; //!< First expression
    E2 m_expr2; //!< Second expression
    letter_expr<K> m_ewidx; //!< Shared indexes
    sequence<N + M + K, const letter*> m_defout; //!< Default output label

public:
    /** \brief Creates the expression core
        \param ewidx Letter expression indicating which indexes are
            shared.
        \param expr1 First expression (A).
        \param expr2 Second expression (B).
        \throw expr_exception If letters are inconsistent.
     **/
    ewmult_core(const letter_expr<K> &ewidx,
        const E1 &expr1, const E2 &expr2) throw(expr_exception);

    /** \brief Copy constructor
     **/
    ewmult_core(const ewmult_core<N, M, K, T, E1, E2> &core);

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

    /** \brief Returns the shared indexes
     **/
    const letter_expr<K> &get_ewidx() const {
        return m_ewidx;
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


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
const char *ewmult_core<N, M, K, T, E1, E2>::k_clazz =
    "ewmult_core<N, M, K, T, E1, E2>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
ewmult_core<N, M, K, T, E1, E2>::ewmult_core(
    const letter_expr<K> &ewidx, const E1 &expr1, const E2 &expr2)
    throw(expr_exception):
        m_ewidx(ewidx),
        m_expr1(expr1),
        m_expr2(expr2) {

    static const char *method =
        "ewmult_core(const letter_expr<K>&, const E1&, const E2&)";

    for(size_t i = 0; i < K; i++) {
        const letter &l = ewidx.letter_at(i);
        if(!expr1.contains(l) || !expr2.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method,
                __FILE__, __LINE__,
                "Shared index is absent from arguments.");
        }
    }

    size_t j = 0;
    for(size_t i = 0; i < N + M + K; i++) m_defout[i] = 0;
    for(size_t i = 0; i < N + K; i++) {
        const letter &l = expr1.letter_at(i);
        if(!ewidx.contains(l)) {
            if(expr2.contains(l)) {
                throw expr_exception(g_ns, k_clazz, method,
                    __FILE__, __LINE__,
                    "Duplicate index in A.");
            } else {
                m_defout[j++] = &l;
            }
        }
    }
    for(size_t i = 0; i < M + K; i++) {
        const letter &l = expr2.letter_at(i);
        if(!ewidx.contains(l)) {
            if(expr1.contains(l)) {
                throw expr_exception(g_ns, k_clazz, method,
                    __FILE__, __LINE__,
                    "Duplicate index in B.");
            } else {
                m_defout[j++] = &l;
            }
        }
    }
    for(size_t i = 0; i < N + K; i++) {
        const letter &l = expr1.letter_at(i);
        if(ewidx.contains(l)) m_defout[j++] = &l;
    }
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
ewmult_core<N, M, K, T, E1, E2>::ewmult_core(
    const ewmult_core<N, M, K, T, E1, E2> &core) :
    m_expr1(core.m_expr1), m_expr2(core.m_expr2), m_ewidx(core.m_ewidx),
    m_defout(core.m_defout) {

}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline bool ewmult_core<N, M, K, T, E1, E2>::contains(const letter &let) const {

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline size_t ewmult_core<N, M, K, T, E1, E2>::index_of(
    const letter &let) const throw(expr_exception) {

    static const char *method = "index_of(const letter&)";

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Expression doesn't contain the letter.");
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline const letter &ewmult_core<N, M, K, T, E1, E2>::letter_at(size_t i) const
    throw(out_of_bounds) {

    static const char *method = "letter_at(size_t)";

    if(i >= N + M + K) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_CORE_H
