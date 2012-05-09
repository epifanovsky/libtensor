#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_DIRSUM_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_DIRSUM_H

#include "../../defs.h"
#include "../../exception.h"
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, typename T, typename E1, typename E2>
class eval_dirsum;


/** \brief Direct sum operation expression core
    \tparam N Order of the first %tensor (A).
    \tparam M Order of the second %tensor (B).
    \tparam E1 First expression (A) type.
    \tparam E2 Second expression (B) type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, typename E1, typename E2>
class core_dirsum {
public:
    static const char *k_clazz; //!< Class name

public:
     //!    Evaluating container type
    typedef eval_dirsum<N, M, T, E1, E2> eval_container_t;

private:
    E1 m_expr1; //!< First expression
    E2 m_expr2; //!< Second expression
    const letter *m_defout[N + M]; //!< Default output label

public:
    /** \brief Creates the expression core
        \param expr1 First expression (A).
        \param expr2 Second expression (B).
        \throw expr_exception If letters are inconsistent.
     **/
    core_dirsum(const E1 &expr1, const E2 &expr2) throw(expr_exception);

    /** \brief Copy constructor
     **/
    core_dirsum(const core_dirsum<N, M, T, E1, E2> &core);

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


template<size_t N, size_t M, typename T, typename E1, typename E2>
const char *core_dirsum<N, M, T, E1, E2>::k_clazz =
    "core_dirsum<N, M, T, E1, E2>";


template<size_t N, size_t M, typename T, typename E1, typename E2>
core_dirsum<N, M, T, E1, E2>::core_dirsum(
    const E1 &expr1, const E2 &expr2) throw(expr_exception) :

    m_expr1(expr1), m_expr2(expr2) {

    static const char *method = "core_dirsum(const E1&, const E2&)";

    size_t j = 0;
    for(size_t i = 0; i < N + M; i++) m_defout[i] = 0;
    for(size_t i = 0; i < N; i++) {
        const letter &l = expr1.letter_at(i);
        if(expr2.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method,
                __FILE__, __LINE__,
                "Duplicate uncontracted index in A.");
        } else {
            m_defout[j++] = &l;
        }
    }
    for(size_t i = 0; i < M; i++) {
        const letter &l = expr2.letter_at(i);
        if(expr1.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method,
                __FILE__, __LINE__,
                "Duplicate uncontracted index in B.");
        } else {
            m_defout[j++] = &l;
        }
    }
}


template<size_t N, size_t M, typename T, typename E1, typename E2>
core_dirsum<N, M, T, E1, E2>::core_dirsum(
    const core_dirsum<N, M, T, E1, E2> &core) :

    m_expr1(core.m_expr1), m_expr2(core.m_expr2) {

    for(size_t i = 0; i < N + M; i++) {
        m_defout[i] = core.m_defout[i];
    }
}


template<size_t N, size_t M, typename T, typename E1, typename E2>
inline bool core_dirsum<N, M, T, E1, E2>::contains(const letter &let) const {

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N, size_t M, typename T, typename E1, typename E2>
inline size_t core_dirsum<N, M, T, E1, E2>::index_of(
    const letter &let) const throw(expr_exception) {

    static const char *method = "index_of(const letter&)";

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Expression doesn't contain the letter.");
}


template<size_t N, size_t M, typename T, typename E1, typename E2>
inline const letter &core_dirsum<N, M, T, E1, E2>::letter_at(size_t i) const
    throw(out_of_bounds) {

    static const char *method = "letter_at(size_t)";

    if(i >= N + M) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_DIRSUM_H
