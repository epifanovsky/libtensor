#ifndef LIBTENSOR_IFACE_SYMM3_CORE_H
#define LIBTENSOR_IFACE_SYMM3_CORE_H

#include <libtensor/exception.h>
#include <libtensor/block_tensor/btod_symmetrize3.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../iface.h" // for g_ns

namespace libtensor {
namespace iface {


/** \brief Expression core for the symmetrization over three indexes
    \tparam N Tensor order.

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class symm3_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    sequence<3, size_t> m_sym;
    expr_core_ptr<N, T> m_subexpr; //!< Sub-expression
    bool m_do_symm; //!< Do symmetrize

public:
    /** \brief Creates the expression core
        \param sym Symmetrized indexes.
        \param subexpr Sub-expression.
        \param do_symm Symmetrize / anti-symmetrize.
     **/
    symm3_core(
        const sequence<3, size_t> &sym,
        const expr_core_ptr<N, T> &subexpr,
        bool do_symm);

    /** \brief Virtual destructor
     **/
    virtual ~symm3_core() { }

    /** \brief Returns the first symmetrized index
     **/
    sequence<3, size_t> &get_sym() {
        return m_sym;
    }

    /** \brief Returns the first symmetrized index
     **/
    const sequence<3, size_t> &get_sym() const {
        return m_sym;
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



template<size_t N, typename T>
const char symm3_core<N, T>::k_clazz[] = "symm3_core<N, T>";


template<size_t N, typename T>
symm3_core<N, T>::symm3_core(
    const sequence<3, size_t> &sym,
    const expr_core_ptr<N, T> &subexpr,
    bool do_symm) :

    m_sym(sym), m_subexpr(subexpr), m_do_symm(do_symm) {

    static const char method[] = "symm3_core(const sequence<3, size_t> &, "
        "const expr_core_ptr<N, T>&, bool)";

    if (m_sym[0] == m_sym[1] || m_sym[0] == m_sym[2] || m_sym[1] == m_sym[2]) {
        throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Symmetrized indexes must be different.");
    }
    if (m_sym[0] < N || m_sym[1] < N || m_sym[2] < N) {
        throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Invalid symmetrized index.");
    }

}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_SYMM3_CORE_H
