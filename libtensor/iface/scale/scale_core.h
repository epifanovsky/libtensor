#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SCALE_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SCALE_CORE_H

#include <libtensor/core/noncopyable.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Expression core that scales an underlying expression
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class scale_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    T m_coeff; //!< Scaling coefficient
    expr_rhs<N, T> m_expr; //!< Unscaled expression

public:
    /** \brief Constructs the scaling expression using a coefficient
            and the underlying unscaled expression
     **/
    scale_core(const T &coeff, const expr_rhs<N, T> &subexpr) :
        m_coeff(coeff), m_expr(subexpr)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~scale_core() { }

    /** \brief Clones this object using new
     **/
    virtual expr_core_i<N, T> *clone() const {
        return new scale_core(*this);
    }

    /** \brief Returns the unscaled expression
     **/
    expr_rhs<N, T> &get_unscaled_expr() {
        return m_expr;
    }

    /** \brief Returns the unscaled expression (const version)
     **/
    const expr_rhs<N, T> &get_unscaled_expr() const {
        return m_expr;
    }

    /** \brief Returns the scaling coefficient
     **/
    const T &get_coeff() {
        return m_coeff;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

    /** \brief Returns whether the tensor's label contains a letter
     **/
    virtual bool contains(const letter &let) const {
        return m_expr.get_core().contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    virtual size_t index_of(const letter &let) const {
        return m_expr.get_core().index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    virtual const letter &letter_at(size_t i) const {
        return m_expr.get_core().letter_at(i);
    }

};


/** \brief Evaluates a scaled expression
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class scale_eval : public eval_container_i<N, T>, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    scale_core<N, T> m_core; //!< Expression core
    std::auto_ptr< eval_container_i<N, T> >
        m_unscaled_cont; //!< Original expression

public:
    /** \brief Constructs the evaluating container
     **/
    scale_eval(
        const scale_core<N, T> &core,
        const letter_expr<N> &label);

    /** \brief Virtual destructor
     **/
    virtual ~scale_eval() { }

    /** \brief Evaluates sub-expressions into temporary tensors
     **/
    virtual void prepare();

    /** \brief Cleans up temporary tensors
     **/
    virtual void clean();

    /** \brief Returns the number of tensors in expression
     **/
    virtual size_t get_ntensor() const;

    /** \brief Returns the number of tensor operations in expression
     **/
    virtual size_t get_noper() const;

    /** \brief Returns tensor arguments
        \param i Argument number.
     **/
//    virtual arg<N, T, tensor_tag> get_tensor_arg(size_t i);

    /** \brief Returns operation arguments
        \param i Argument number.
     **/
//    virtual arg<N, T, oper_tag> get_oper_arg(size_t i);

};


template<size_t N, typename T>
const char scale_eval<N, T>::k_clazz[] = "scale_eval<N, T>";


template<size_t N, typename T>
scale_eval<N, T>::scale_eval(
    const scale_core<N, T> &core,
    const letter_expr<N> &label) :

    m_core(core),
    m_unscaled_cont(m_core.get_unscaled_expr().get_core().
        create_container(label)) {

}


template<size_t N, typename T>
void scale_eval<N, T>::prepare() {

    m_unscaled_cont->prepare();
}


template<size_t N, typename T>
void scale_eval<N, T>::clean() {

    m_unscaled_cont->clean();
}


template<size_t N, typename T>
size_t scale_eval<N, T>::get_ntensor() const {

    return m_unscaled_cont->get_ntensor();
}


template<size_t N, typename T>
size_t scale_eval<N, T>::get_noper() const {

    return m_unscaled_cont->get_noper();
}


//template<size_t N, typename T>
//arg<N, T, tensor_tag> scale_eval<N, T>::get_tensor_arg(size_t i) {
//
//    arg<N, T, tensor_tag> argument = m_unscaled_cont->get_tensor_arg(i);
//    argument.scale(m_core.get_coeff());
//    return argument;
//}
//
//
//template<size_t N, typename T>
//arg<N, T, oper_tag> scale_eval<N, T>::get_oper_arg(size_t i) {
//
//    arg<N, T, oper_tag> argument = m_unscaled_cont->get_oper_arg(i);
//    argument.scale(m_core.get_coeff());
//    return argument;
//}


template<size_t N, typename T>
eval_container_i<N, T> *scale_core<N, T>::create_container(
    const letter_expr<N> &label) const {

    return new scale_eval<N, T>(*this, label);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SCALE_CORE_H
