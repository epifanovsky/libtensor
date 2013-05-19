#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EXPR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EXPR_H

#include <libtensor/exception.h>
#include <libtensor/core/noncopyable.h>
#include "../labeled_btensor.h"
#include "arg.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Evaluation container base class

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class eval_container_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~eval_container_i() { }

    /** \brief Prepares the container
     **/
    virtual void prepare() = 0;

    /** \brief Cleans up the container
     **/
    virtual void clean() = 0;

    /** \brief Returns the number of tensors in expression
     **/
    virtual size_t get_ntensor() const = 0;

    /** \brief Returns the number of tensor operations in expression
     **/
    virtual size_t get_noper() const = 0;

    /** \brief Returns tensor arguments
        \param i Argument number.
     **/
    virtual arg<N, T, tensor_tag> get_tensor_arg(size_t i) = 0;

    /** \brief Returns operation arguments
        \param i Argument number.
     **/
    virtual arg<N, T, oper_tag> get_oper_arg(size_t i) = 0;

};


/** \brief Expression core base class

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class expr_core_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~expr_core_i() { }

    /** \brief Clones this expression core
     **/
    virtual expr_core_i<N, T> *clone() const = 0;

    /** \brief Creates an evaluation container using new, caller responsible
            to call delete
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const = 0;

    /** \brief Returns whether the label contains a letter
     **/
    virtual bool contains(const letter &let) const = 0;

    /** \brief Returns the index of a letter in the label
     **/
    virtual size_t index_of(const letter &let) const = 0;

    /** \brief Returns the letter at a given position in the label
     **/
    virtual const letter &letter_at(size_t i) const = 0;

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
template<size_t N, typename T>
class expr : public noncopyable {
private:
    expr_core_i<N, T> *m_core; //!< Expression core

public:
    /** \brief Constructs the expression using a core
     **/
    expr(const expr_core_i<N, T> &core) : m_core(core.clone()) { }

    /** \brief Copy constructor
     **/
    expr(const expr<N, T> &expr) : m_core(expr.m_core->clone()) { }

    /** \brief Virtual destructor
     **/
    virtual ~expr() {
        delete m_core;
    }

    /** \brief Returns the core of the expression
     **/
    expr_core_i<N, T> &get_core() {
        return *m_core;
    }

    /** \brief Returns the core of the expression (const version)
     **/
    const expr_core_i<N, T> &get_core() const {
        return *m_core;
    }

#if 0
    /** \brief Returns whether the label contains a letter
     **/
    bool contains(const letter &let) const {
        return m_core->contains(let);
    }

    /** \brief Returns the %index of a %letter in the label
     **/
    size_t index_of(const letter &let) const {
        return m_core->index_of(let);
    }

    /** \brief Returns the %letter at a given position in the label
     **/
    const letter &letter_at(size_t i) const {
        return m_core->letter_at(i);
    }
#endif

};


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EXPR_H

