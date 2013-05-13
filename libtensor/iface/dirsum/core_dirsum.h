#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_DIRSUM_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_DIRSUM_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, typename T> class eval_dirsum;


/** \brief Direct sum operation expression core
    \tparam N Order of the first tensor (A).
    \tparam M Order of the second tensor (B).

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class core_dirsum : public expr_core_i<N + M, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
     //! Evaluating container type
    typedef eval_dirsum<N, M, T> eval_container_t;

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
    core_dirsum(const expr<N, T> &expr1, const expr<M, T> &expr2)

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


template<size_t N, size_t M, typename T>
const char core_dirsum<N, M, T>::k_clazz[] = "core_dirsum<N, M, T>";


template<size_t N, size_t M, typename T>
core_dirsum<N, M, T>::core_dirsum(
    const expr<N, T> &expr1, const expr<M, T> &expr2) :

    m_expr1(expr1), m_expr2(expr2), m_defout(0) {

    static const char method[] =
        "core_dirsum(const expr<N, T>&, const expr<M, T>&)";

    size_t j = 0;
    for(size_t i = 0; i < N; i++) {
        const letter &l = expr1.letter_at(i);
        if(expr2.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Duplicate uncontracted index in A.");
        } else {
            m_defout[j++] = &l;
        }
    }
    for(size_t i = 0; i < M; i++) {
        const letter &l = expr2.letter_at(i);
        if(expr1.contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Duplicate uncontracted index in B.");
        } else {
            m_defout[j++] = &l;
        }
    }
}


template<size_t N, size_t M, typename T>
bool core_dirsum<N, M, T>::contains(const letter &let) const {

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return true;
    }
    return false;
}


template<size_t N, size_t M, typename T>
size_t core_dirsum<N, M, T>::index_of(const letter &let) const {

    static const char method[] = "index_of(const letter&)";

    for(register size_t i = 0; i < N + M; i++) {
        if(m_defout[i] == &let) return i;
    }

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Expression doesn't contain the letter.");
}


template<size_t N, size_t M, typename T>
const letter &core_dirsum<N, M, T>::letter_at(size_t i) const {

    static const char method[] = "letter_at(size_t)";

    if(i >= N + M) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Letter index is out of bounds.");
    }
    return *(m_defout[i]);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_DIRSUM_H
