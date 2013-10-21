#ifndef LIBTENSOR_IFACE_ADD_CORE_H
#define LIBTENSOR_IFACE_ADD_CORE_H

#include <libtensor/core/permutation.h>
#include "../expr_core_i.h"

namespace libtensor {
namespace iface {

/** \brief Addition operation expression core
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class add_core : public expr_core_i<N, T> {
private:
    expr_core_ptr<N, T> m_expr1; //!< First expression
    expr_core_ptr<N, T> m_expr2; //!< Second expression
    permutation<N> m_perm; //!< Permutation of second expression

public:
    /** \brief Initializes the core with first and second expressions
        \param expr1 First expression
        \param expr2 Second expression
        \param perm Permutation of second expression
     **/
    add_core(
        const expr_core_ptr<N, T> &expr1,
        const expr_core_ptr<N, T> &expr2,
        const permutation<N> &perm)  :
        m_expr1(expr1), m_expr2(expr2), m_perm(perm) {
    }

    /** \brief Virtual destructor
     **/
    virtual ~add_core() { }

    /** \brief Returns type of the expression
     **/
    virtual const std::string &get_type() const {
        return "add";
    }

    /** \brief Returns the left expression
     **/
    expr_core_ptr<N, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the left expression, const version
     **/
    const expr_core_ptr<N, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the right expression
     **/
    expr_core_ptr<N, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the right expression
     **/
    const expr_core_ptr<N, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Permute the right expression
     **/
    void permute(const permutation<N> &perm) {
        m_perm.permute(perm);
    }

    /** \brief Return permutation of right expression
     **/
    const permutation<N> &get_perm() {
        return m_perm;
    }
};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_ADD_CORE_H
