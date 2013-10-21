#ifndef LIBTENSOR_IFACE_DIAG_CORE_H
#define LIBTENSOR_IFACE_DIAG_CORE_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace iface {


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
    sequence<M, size_t> m_diag; //!< Diagonal indexes
    expr_core_ptr<N, T> m_subexpr; //!< Sub-expression

public:
    /** \brief Creates the expression core
        \param diag_letter Letter in the output.
        \param diag_label Expression defining the diagonal.
        \param subexpr Sub-expression.
     **/
    diag_core(const sequence<M, size_t> &diag,
            const expr_core_ptr<N, T> &subexpr);

    /** \brief Virtual destructor
     **/
    virtual ~diag_core() { }

    /** \brief Returns type of the expression
     **/
    virtual const std::string &get_type() const {
        return "diag";
    }

    /** \brief Return diagonal indexes
     **/
    sequence<M, size_t> &get_diag_indexes() {
        return m_diag;
    }

    /** \brief Return diagonal indexes, const version
     **/
    const sequence<M, size_t> &get_diag_indexes() const {
        return m_diag;
    }

    /** \brief Returns the sub-expression
     **/
    expr_core_ptr<N, T> &get_sub_expr() {
        return m_subexpr;
    }
    /** \brief Returns the sub-expression, const version
     **/
    const expr_core_ptr<N, T> &get_sub_expr() const {
        return m_subexpr;
    }
};


template<size_t N, size_t M, typename T>
const char diag_core<N, M, T>::k_clazz[] = "diag_core<N, M, T>";


template<size_t N, size_t M, typename T>
diag_core<N, M, T>::diag_core(
    const sequence<M, size_t> &diag,
    const expr_core_ptr<N, T> &subexpr) :
    m_diag(diag), m_subexpr(subexpr) {

    for (size_t i = 0; i < M; i++) {
        if (m_diag[i] < N) continue;

        throw expr_exception(g_ns, k_clazz,
                "diag_core()", __FILE__, __LINE__, "diag");
    }
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_IFACE_DIAG_CORE_H
