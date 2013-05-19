#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_CORE_H

#include <libtensor/core/noncopyable.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Identity expression core (references one labeled tensor)
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Assignable Whether the tensor is an l-value.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable>
class ident_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    //! Labeled block tensor type
    typedef labeled_btensor<N, T, Assignable> labeled_btensor_t;

private:
    labeled_btensor_t m_t; //!< Labeled block tensor

public:
    /** \brief Initializes the operation with a tensor reference
     **/
    ident_core(const labeled_btensor_t &t) : m_t(t) { }

    /** \brief Virtual destructor
     **/
    virtual ~ident_core() { }

    /** \brief Clones this object using new
     **/
    virtual expr_core_i<N, T> *clone() const {
        return new ident_core(*this);
    }

    /** \brief Returns the labeled block tensor
     **/
    labeled_btensor_t &get_tensor() {
        return m_t;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

    /** \brief Returns whether the tensor's label contains a letter
     **/
    virtual bool contains(const letter &let) const {
        return m_t.contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    virtual size_t index_of(const letter &let) const {
        return m_t.index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    virtual const letter &letter_at(size_t i) const {
        return m_t.letter_at(i);
    }

};


template<size_t N, typename T, bool Assignable>
const char ident_core<N, T, Assignable>::k_clazz[] =
    "ident_core<N, T, Assignable>";


/** \brief Evaluating container for a labeled tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam A Whether the tensor is an l-value.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool A>
class ident_eval : public eval_container_i<N, T>, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    ident_core<N, T, A> m_core;
    permutation<N> m_perm;

public:
    /** \brief Constructs the evaluating container
     **/
    ident_eval(
        const ident_core<N, T, A> &core,
        const letter_expr<N> &label);

    /** \brief Virtual destructor
     **/
    virtual ~ident_eval() { }

    /** \brief Prepares the container
     **/
    virtual void prepare() { }

    /** \brief Cleans up the container
     **/
    virtual void clean() { }

    /** \brief Returns the number of tensors in expression
     **/
    virtual size_t get_ntensor() const {
        return 1;
    }

    /** \brief Returns the number of tensor operations in expression
     **/
    virtual size_t get_noper() const {
        return 0;
    }

    /** \brief Returns the tensor argument
        \param i Argument number (0 is the only allowed value).
     **/
    virtual arg<N, T, tensor_tag> get_tensor_arg(size_t i);

    /** \brief Returns operation arguments (not valid for ident_eval)
        \param i Argument number.
     **/
    virtual arg<N, T, oper_tag> get_oper_arg(size_t i);

};


template<size_t N, typename T, bool A>
const char ident_eval<N, T, A>::k_clazz[] = "ident_eval<N, T, A>";


template<size_t N, typename T, bool A>
ident_eval<N, T, A>::ident_eval(
    const ident_core<N, T, A> &core,
    const letter_expr<N> &label) :

    m_core(core),
    m_perm(label.permutation_of(m_core.get_tensor().get_label())) {

}


template<size_t N, typename T, bool A>
arg<N, T, tensor_tag> ident_eval<N, T, A>::get_tensor_arg(size_t i) {

    static const char method[] = "get_tensor_arg(size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

    return arg<N, T, tensor_tag>(
        m_core.get_tensor().get_btensor(), m_perm, 1.0);
}


template<size_t N, typename T, bool A>
arg<N, T, oper_tag> ident_eval<N, T, A>::get_oper_arg(size_t i) {

    static const char method[] = "get_oper_arg(size_t)";

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, typename T, bool A>
eval_container_i<N, T> *ident_core<N, T, A>::create_container(
    const letter_expr<N> &label) const {

    return new ident_eval<N, T, A>(*this, label);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_CORE_H
