#ifndef LIBTENSOR_IFACE_DIRSUM_CORE_H
#define LIBTENSOR_IFACE_DIRSUM_CORE_H

#include <libtensor/exception.h>
#include <libtensor/core/sequence.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace iface {


/** \brief Direct sum operation expression core
    \tparam N Order of the first tensor (A).
    \tparam M Order of the second tensor (B).

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class dirsum_core : public expr_core_i<N + M, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    expr_rhs<N, T> m_expr1; //!< First expression
    expr_rhs<M, T> m_expr2; //!< Second expression

public:
    /** \brief Creates the expression core
        \param expr1 First expression (A).
        \param expr2 Second expression (B).
     **/
    dirsum_core(
        const expr_core_ptr<N, T> &expr1,
        const expr_core_ptr<M, T> &expr2) :
        m_expr1(expr1), m_expr2(expr2) {

    }

    /** \brief Virtual destructor
     **/
    virtual ~dirsum_core() { }

    /** \brief Returns type of the expression
     **/
    virtual const std::string &get_type() const {
        return "dirsum";
    }

    /** \brief Returns the first expression (A)
     **/
    expr_core_ptr<N, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (A), const version
     **/
    const expr_core_ptr<N, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression (B)
     **/
    expr_core_ptr<M, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (B), const version
     **/
    const expr_core_ptr<M, T> &get_expr_2() const {
        return m_expr2;
    }
};

} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_DIRSUM_CORE_H
