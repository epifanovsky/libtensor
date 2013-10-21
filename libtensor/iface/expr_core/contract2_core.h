#ifndef LIBTENSOR_IFACE_CONTRACT2_CORE_H
#define LIBTENSOR_IFACE_CONTRACT2_CORE_H

#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace iface {


/** \brief Two-tensor contraction operation expression core
    \tparam N Order of the first tensor (A) less contraction degree.
    \tparam M Order of the second tensor (B) less contraction degree.
    \tparam K Number of indexes contracted.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class contract2_core : public expr_core_i<N + M, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    sequence<2 * K, size_t> m_contr;
    expr_core_ptr<N + K, T> m_expr1; //!< First expression
    expr_core_ptr<M + K, T> m_expr2; //!< Second expression

public:
    /** \brief Creates the expression core
        \param contr Letter expression indicating which indexes will be
            contracted.
        \param expr1 First expression (A).
        \param expr2 Second expression (B).
        \throw expr_exception If letters are inconsistent.
     **/
    contract2_core(
        const sequence<2 * K, size_t> &contr,
        const expr_core_ptr<N + K, T> &expr1,
        const expr_core_ptr<M + K, T> &expr2);

    /** \brief Virtual destructor
     **/
    virtual ~contract2_core() { }

    /** \brief Returns type of the expression
     **/
    virtual const std::string &get_type() const {
        return "contract2";
    }

    /** \brief Returns the first expression (A)
     **/
    expr_core_ptr<N + K, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (A), const version
     **/
    const expr_core_ptr<N + K, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression (B)
     **/
    expr_core_ptr<M + K, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (B), const version
     **/
    const expr_core_ptr<M + K, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Returns the contracted indexes
     **/
    sequence<2 * K, size_t> &get_contr() {
        return m_contr;
    }

    /** \brief Returns the contracted indexes
     **/
    const sequence<2 * K, size_t> &get_contr() const {
        return m_contr;
    }
};


template<size_t N, size_t M, size_t K, typename T>
const char contract2_core<N, M, K, T>::k_clazz[] = "contract2_core<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
contract2_core<N, M, K, T>::contract2_core(
    const sequence<2 * K, size_t> &contr,
    const expr_core_ptr<N + K, T> &expr1,
    const expr_core_ptr<M + K, T> &expr2) :
    m_contr(contr), m_expr1(expr1), m_expr2(expr2) {


    for (size_t i = 0; i < K; i++) {
        if (m_contr[i] < N + K && m_contr[i + K] < M + K) continue;

        if (m_contr[i] < N + K)
            throw expr_exception(g_ns, k_clazz,
                    "contract2_core()", __FILE__, __LINE__, "contr (2)");
        else
            throw expr_exception(g_ns, k_clazz,
                    "contract2_core()", __FILE__, __LINE__, "contr (1)");
    }
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_CONTRACT2_CORE_H
