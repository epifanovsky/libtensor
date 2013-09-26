#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_CORE_H

#include <libtensor/core/noncopyable.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns
#include "../any_tensor.h"

namespace libtensor {
namespace labeled_btensor_expr {
using namespace libtensor::iface;


/** \brief Identity expression core (references one labeled tensor)
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class ident_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    any_tensor<N, T> m_t; //!< Tensor
    letter_expr<N> m_label; //!< Letter label

public:
    /** \brief Initializes the operation with a tensor reference
     **/
    ident_core(const any_tensor<N, T> &t, const letter_expr<N> &label) :
        m_t(t), m_label(label)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~ident_core() { }

    /** \brief Clones this object using new
     **/
    virtual expr_core_i<N, T> *clone() const {
        return new ident_core(*this);
    }

    /** \brief Returns the enclosed tensor
     **/
    const any_tensor<N, T> &get_tensor() const {
        return m_t;
    }

    /** \brief Returns the label
     **/
    const letter_expr<N> &get_label() const {
        return m_label;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

    /** \brief Returns whether the tensor's label contains a letter
     **/
    virtual bool contains(const letter &let) const {
        return m_label.contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    virtual size_t index_of(const letter &let) const {
        return m_label.index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    virtual const letter &letter_at(size_t i) const {
        return m_label.letter_at(i);
    }

};


template<size_t N, typename T>
const char ident_core<N, T>::k_clazz[] = "ident_core<N, T>";


/** \brief Evaluating container for a labeled tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class ident_eval : public eval_container_i<N, T>, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    ident_core<N, T> m_core;
    permutation<N> m_perm;

public:
    /** \brief Constructs the evaluating container
     **/
    ident_eval(
        const ident_core<N, T> &core,
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
//    virtual arg<N, T, tensor_tag> get_tensor_arg(size_t i);

    /** \brief Returns operation arguments (not valid for ident_eval)
        \param i Argument number.
     **/
//    virtual arg<N, T, oper_tag> get_oper_arg(size_t i);

};


template<size_t N, typename T>
const char ident_eval<N, T>::k_clazz[] = "ident_eval<N, T>";


template<size_t N, typename T>
ident_eval<N, T>::ident_eval(
    const ident_core<N, T> &core,
    const letter_expr<N> &label) :

    m_core(core),
    m_perm(label.permutation_of(m_core.get_label())) {

}


//template<size_t N, typename T>
//arg<N, T, tensor_tag> ident_eval<N, T>::get_tensor_arg(size_t i) {
//
//    static const char method[] = "get_tensor_arg(size_t)";
//
//    if(i != 0) {
//        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
//            "Argument index is out of bounds.");
//    }
//
//    return arg<N, T, tensor_tag>(
//        m_core.get_tensor().get_btensor(), m_perm, 1.0);
//}
//
//
//template<size_t N, typename T>
//arg<N, T, oper_tag> ident_eval<N, T>::get_oper_arg(size_t i) {
//
//    static const char method[] = "get_oper_arg(size_t)";
//
//    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
//        "Invalid method.");
//}


template<size_t N, typename T>
eval_container_i<N, T> *ident_core<N, T>::create_container(
    const letter_expr<N> &label) const {

    return new ident_eval<N, T>(*this, label);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_CORE_H
