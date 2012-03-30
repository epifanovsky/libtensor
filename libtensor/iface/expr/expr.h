#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EXPR_H
#define    LIBTENSOR_LABELED_BTENSOR_EXPR_EXPR_H

#include "../../defs.h"
#include "../../exception.h"
#include "../labeled_btensor.h"
#include "arg.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Expression base class

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class expr_i {
public:
    virtual ~expr_i() { }
};


/** \brief Expression meta-wrapper
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Core Expression core type.

    Tensor expressions make extensive use of a meta-programming technique
    call "expression templates". It allows us to store the expression
    tree as the C++ type thus transferring a number of sanity checks to
    the compilation level.

    This template wraps around the real expression type (core) to facilitate
    the matching of overloaded operator templates.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Core>
class expr : public expr_i<N, T> {
public:
    //!    Expression core type
    typedef Core core_t;

    //!    Expression evaluating container type
    typedef typename Core::eval_container_t eval_container_t;

private:
    Core m_core; //!< Expression core

public:
    /** \brief Constructs the expression using a core
     **/
    expr(const Core &core) : m_core(core) { }

    /** \brief Copy constructor
     **/
    expr(const expr<N, T, Core> &expr) : m_core(expr.m_core) { }

    /** \brief Virtual destructor
     **/
    virtual ~expr() { }

    /** \brief Returns the core of the expression
     **/
    Core &get_core();

    /** \brief Returns the core of the expression (const version)
     **/
    const Core &get_core() const;

    /** \brief Returns whether the label contains a %letter
     **/
    bool contains(const letter &let) const;

    /** \brief Returns the %index of a %letter in the label
     **/
    size_t index_of(const letter &let) const throw(exception);

    /** \brief Returns the %letter at a given position in the label
     **/
    const letter &letter_at(size_t i) const throw(exception);

private:
    expr<N, T, Core> &operator=(const expr<N, T, Core>&);
};


template<size_t N, typename T, typename Core>
inline Core &expr<N, T, Core>::get_core() {

    return m_core;
}


template<size_t N, typename T, typename Core>
inline const Core &expr<N, T, Core>::get_core() const {

    return m_core;
}


template<size_t N, typename T, typename Core>
inline bool expr<N, T, Core>::contains(const letter &let) const {

    return m_core.contains(let);
}


template<size_t N, typename T, typename Core>
inline size_t expr<N, T, Core>::index_of(const letter &let) const
    throw(exception) {

    return m_core.index_of(let);
}


template<size_t N, typename T, typename Core>
inline const letter &expr<N, T, Core>::letter_at(size_t i) const
    throw(exception) {

    return m_core.letter_at(i);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EXPR_H

