#ifndef LIBTENSOR_IFACE_SYMM2_CORE_H
#define LIBTENSOR_IFACE_SYMM2_CORE_H

#include "../expr_exception.h"

namespace libtensor {
namespace iface {


/** \brief Expression core for the symmetrization over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in the set.
    \tparam Sym Symmetrization/antisymmetrization.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class symm2_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    sequence<2 * M, size_t> m_sym; //!< Symmetrized indexes
    expr_core_ptr<N, T> m_subexpr; //!< Sub-expression
    bool m_do_symm; //!< Symmetrize

public:
    /** \brief Creates the expression core
        \param sym Sequence indicating symmetrized indexes
        \param subexpr Sub-expression
        \param symm Symmetrize or anti-symmetrize
     **/
    symm2_core(const sequence<2 * M, size_t> &sym,
        const expr_core_ptr<N, T> &subexpr, bool do_symm);

    /** \brief Virtual destructor
     **/
    virtual ~symm2_core() { }

    /** \brief Returns the set of symmetrized indexes
     **/
    sequence<2 * M, size_t> &get_sym() {
        return m_sym;
    }

    /** \brief Returns the set of symmetrized indexes, const version
     **/
    const sequence<2 * M, size_t> &get_sym() const {
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


template<size_t N, size_t M, typename T>
const char symm2_core<N, M, T>::k_clazz[] = "symm2_core<N, M, T>";


template<size_t N, size_t M, typename T>
symm2_core<N, M, T>::symm2_core(const sequence<2 * M, size_t> &sym,
    const expr_core_ptr<N, T> &subexpr, bool do_symm) :
    m_sym(sym), m_subexpr(subexpr), m_do_symm(do_symm)  {

    static const char method[] = "symm2_core(const sequence<2*M, size_t>&, "
        "const expr_core_ptr<N, T>&, bool)";

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < M; j++) {
            if (m_sym[i] == m_sym[j + M]) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Symmetrized indexes must be different.");
            }
            if (i != j && m_sym[i] == m_sym[j]) {
                throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Indexes can only appear once.");
            }
        }

        if (m_sym[i] < N && m_sym[i + M] < N)  continue;
        throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Unknown index");
        }
    }
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_SYMM2_CORE_H
