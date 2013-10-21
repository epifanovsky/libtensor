#ifndef LIBTENSOR_IFACE_MULT_CORE_H
#define LIBTENSOR_IFACE_MULT_CORE_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace iface {


/** \brief Element-wise multiplication operation expression core
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class mult_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    expr_core_ptr<N, T> m_expr1; //!< Left expression
    expr_core_ptr<N, T> m_expr2; //!< Right expression
    permutation<N> m_perm; //!< Permutation
    bool m_recip; //!< Do element-wise division

public:
    /** \brief Initializes the core with left and right expressions
     **/
    mult_core(
        const expr_core_ptr<N, T> &expr1,
        const expr_core_ptr<N, T> &expr2,
        const permutation<N> &perm,
        bool recip);

    /** \brief Virtual destructor
     **/
    virtual ~mult_core() { }

    /** \brief Returns type of the expression
     **/
    virtual const std::string &get_type() const {
        return "mult";
    }

    /** \brief Returns the first expression
     **/
    expr_core_ptr<N, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (const version)
     **/
    const expr_core_ptr<N, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression
     **/
    expr_core_ptr<N, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (const version)
     **/
    const expr_core_ptr<N, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Returns the permutation of the second expression
     **/
    permutation<N> &get_perm() {
        return m_perm;
    }

    /** \brief Returns the permutation of the second expression (const version)
     **/
    const permutation<N, T> &get_perm() const {
        return m_perm;
    }

    /** \brief Perform division
     **/
    bool do_recip() const {
        return m_recip;
    }
};


} // namespace iface
} // namespace libtensor


#endif // LIBTENSOR_IFACE_MULT_CORE_H
