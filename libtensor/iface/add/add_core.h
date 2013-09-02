#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_CORE_H

#include <memory>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Addition operation expression core
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class add_core : public expr_core_i<N, T> {
private:
    expr<N, T> m_expr_l; //!< Left expression
    expr<N, T> m_expr_r; //!< Right expression

public:
    /** \brief Initializes the core with left and right expressions
     **/
    add_core(const expr<N, T> &expr_l, const expr<N, T> &expr_r) :
        m_expr_l(expr_l), m_expr_r(expr_r)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~add_core() { }

    /** \brief Clones this object using new
     **/
    virtual expr_core_i<N, T> *clone() const {
        return new add_core(*this);
    }

    /** \brief Returns the left expression
     **/
    const expr<N, T> &get_expr_l() {
        return m_expr_l;
    }

    /** \brief Returns the right expression
     **/
    const expr<N, T> &get_expr_r() {
        return m_expr_r;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

    /** \brief Returns whether the tensor's label contains a letter
     **/
    virtual bool contains(const letter &let) const {
        return m_expr_l.get_core().contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    virtual size_t index_of(const letter &let) const {
        return m_expr_l.get_core().index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    virtual const letter &letter_at(size_t i) const {
        return m_expr_l.get_core().letter_at(i);
    }

};


/** \brief Evaluates the addition expression
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class add_eval : public eval_container_i<N, T>, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    add_core<N, T> m_core; //!< Expression core
    std::auto_ptr< eval_container_i<N, T> >
        m_cont_l; //!< Left evaluating container
    std::auto_ptr< eval_container_i<N, T> >
        m_cont_r; //!< Right evaluating container

public:
    /** \brief Constructs the evaluating container
     **/
    add_eval(
        const add_core<N, T> &core,
        const letter_expr<N> &label);

    /** \brief Virtual destructor
     **/
    virtual ~add_eval();

    /** \brief Evaluates sub-expressions into temporary tensors
     **/
    void prepare();

    /** \brief Cleans up temporary tensors
     **/
    void clean();

    /** \brief Returns the number of tensors in expression
     **/
    virtual size_t get_ntensor() const;

    /** \brief Returns the number of tensor operations in expression
     **/
    virtual size_t get_noper() const;

    /** \brief Returns tensor arguments
        \param i Argument number.
     **/
    virtual arg<N, T, tensor_tag> get_tensor_arg(size_t i);

    /** \brief Returns operation arguments
        \param i Argument number.
     **/
    virtual arg<N, T, oper_tag> get_oper_arg(size_t i);

};


template<size_t N, typename T>
const char add_eval<N, T>::k_clazz[] = "add_eval<N, T>";


template<size_t N, typename T>
add_eval<N, T>::add_eval(
    const add_core<N, T> &core,
    const letter_expr<N> &label) :

    m_core(core),
    m_cont_l(m_core.get_expr_l().get_core().create_container(label)),
    m_cont_r(m_core.get_expr_r().get_core().create_container(label)) {

}


template<size_t N, typename T>
add_eval<N, T>::~add_eval() {

}


template<size_t N, typename T>
void add_eval<N, T>::prepare() {

    m_cont_l->prepare();
    m_cont_r->prepare();
}


template<size_t N, typename T>
void add_eval<N, T>::clean() {

    m_cont_l->clean();
    m_cont_r->clean();
}


template<size_t N, typename T>
size_t add_eval<N, T>::get_ntensor() const {

    return m_cont_l->get_ntensor() + m_cont_r->get_ntensor();
}


template<size_t N, typename T>
size_t add_eval<N, T>::get_noper() const {

    return m_cont_l->get_noper() + m_cont_r->get_noper();
}


template<size_t N, typename T>
arg<N, T, tensor_tag> add_eval<N, T>::get_tensor_arg(size_t i) {

    size_t nl = m_cont_l->get_ntensor();
    if(i < nl) return m_cont_l->get_tensor_arg(i);
    else return m_cont_r->get_tensor_arg(i - nl);
}


template<size_t N, typename T>
arg<N, T, oper_tag> add_eval<N, T>::get_oper_arg(size_t i) {

    size_t nl = m_cont_l->get_noper();
    if(i < nl) return m_cont_l->get_oper_arg(i);
    else return m_cont_r->get_oper_arg(i - nl);
}


template<size_t N, typename T>
eval_container_i<N, T> *add_core<N, T>::create_container(
    const letter_expr<N> &label) const {

    return new add_eval<N, T>(*this, label);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_CORE_H
