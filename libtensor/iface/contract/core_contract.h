#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_CONTRACT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_CONTRACT_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, size_t K, typename T> class eval_contract;


/** \brief Contraction operation expression core
    \tparam N Order of the first tensor (A) less contraction degree.
    \tparam M Order of the second tensor (B) less contraction degree.
    \tparam K Number of indexes contracted.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class core_contract : public expr_core_i<N + M, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
     //! Evaluating container type
    typedef eval_contract<N, M, K, T> eval_container_t;

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
    core_contract(const letter_expr<K> &contr,
        const expr<N + K, T> &expr1, const expr<M + K, T> &expr2);

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


template<size_t N, size_t M, size_t K, typename T>
const char core_contract<N, M, K, T>::k_clazz[] = "core_contract<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
core_contract<N, M, K, T>::core_contract(
    const letter_expr<K> &contr,
    const expr<N + K, T> &expr1,
    const expr<M + K, T> &expr2) :

    m_contr(contr), m_expr1(expr1), m_expr2(expr2), m_defout(0) {

    static const char method[] = "core_contract(const letter_expr<K>&, "
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
bool core_contract<N, M, K, T>::contains(const letter &let) const {

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N, size_t M, size_t K, typename T>
size_t core_contract<N, M, K, T>::index_of(const letter &let) const {

    static const char method[] = "index_of(const letter&)";

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Expression doesn't contain the letter.");
}


template<size_t N, size_t M, size_t K, typename T>
const letter&core_contract<N, M, K, T>::letter_at(size_t i) const {

    static const char method[] = "letter_at(size_t)";

    if(i >= N + M) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_CONTRACT_H
