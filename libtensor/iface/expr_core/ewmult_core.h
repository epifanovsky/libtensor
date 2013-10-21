#ifndef LIBTENSOR_IFACE_EWMULT_CORE_H
#define LIBTENSOR_IFACE_EWMULT_CORE_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>

namespace libtensor {
namespace iface {

/** \brief Element-wise product operation expression core
    \tparam N Order of the first tensor (A) less number of shared indexes.
    \tparam M Order of the second tensor (B) less number of shared indexes.
    \tparam K Number of shared indexes.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class ewmult_core : public expr_core_i<N + M + K, T> {
public:
    static const char k_clazz[];
public:
    enum {
        NA = N + K,
        NB = M + K,
        NC = N + M + K
    };

private:
    sequence<2 * K, size_t> m_ewidx; //!< Shared indexes
    expr_core_ptr<NA, T> m_expr1; //!< First expression
    expr_core_ptr<NB, T> m_expr2; //!< Second expression

public:
    /** \brief Creates the expression core
        \param ewidx Letter expression indicating which indexes are shared.
        \param expr1 First expression (A).
        \param expr2 Second expression (B).
     **/
    ewmult_core(
        const sequence<2 * K, size_t> &ewidx,
        const expr_core_ptr<NA, T> &expr1,
        const expr_core_ptr<NB, T> &expr2);

    /** \brief Virtual destructor
     **/
    virtual ~ewmult_core() { }

    /** \brief Returns type of the expression
     **/
    virtual const std::string &get_type() const {
        return "ewmult";
    }

    /** \brief Returns the first expression (A)
     **/
    expr_core_ptr<NA, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (A), const version
     **/
    const expr_core_ptr<NA, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression (B)
     **/
    expr_core_ptr<NB, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (B), const version
     **/
    const expr_core_ptr<NB, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Returns the shared indexes
     **/
    sequence<2 * K, size_t> &get_ewidx() {
        return m_ewidx;
    }

    /** \brief Returns the shared indexes
     **/
    const sequence<2 * K, size_t> &get_ewidx() const {
        return m_ewidx;
    }
};


template<size_t N, size_t M, size_t K, typename T>
const char ewmult_core<N, M, K, T>::k_clazz[] = "ewmult_core<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
ewmult_core<N, M, K, T>::ewmult_core(
    const sequence<2 * K, size_t> &ewidx,
    const expr_core_ptr<NA, T> &expr1,
    const expr_core_ptr<NB, T> &expr2) :
    m_ewidx(ewdix), m_expr1(expr1), m_expr2(expr2) {


    for (size_t i = 0; i < K; i++) {
        if (m_ewidx[i] < NA && m_ewidx[i + K] < NB) continue;

        if (m_ewidx[i] < NA) {
            throw expr_exception(g_ns, k_clazz,
                    "ewmult_core()", __FILE__, __LINE__, "ewidx (2)");
        }
        else {
            throw expr_exception(g_ns, k_clazz,
                    "ewmult_core()", __FILE__, __LINE__, "ewidx (1)");
        }
    }
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EWMULT_CORE_H
