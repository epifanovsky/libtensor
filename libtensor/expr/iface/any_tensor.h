#ifndef LIBTENSOR_EXPR_ANY_TENSOR_H
#define LIBTENSOR_EXPR_ANY_TENSOR_H

#include <cstddef> // for size_t
#include <typeinfo>
#include "label.h"

namespace libtensor {
namespace expr {


/** \defgroup libtensor_expr_iface Basic components of programming interface
    \ingroup libtensor_expr
 **/


template<size_t N, typename T>
class expr_rhs;


/** \brief Any tensor type
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This template implements the any concept for tensors. The actual tensor
    type is concealed and is only known to the creator and the recipient.

    \ingroup libtensor_expr_iface
 **/
template<size_t N, typename T>
class any_tensor {
private:
    class holder_base {
    public:
        virtual ~holder_base() { }
        virtual const std::type_info &type_info() const = 0;
        virtual holder_base *clone() const = 0;
    };

    template<typename Tensor>
    class holder : public holder_base {
    public:
        Tensor &m_t;
    public:
        holder(Tensor &t) : m_t(t) { }
        virtual ~holder() { }
        virtual const std::type_info &type_info() const {
            return typeid(Tensor);
        }
        virtual holder_base *clone() const {
            return new holder(m_t);
        }
    };

private:
    holder_base *m_tensor; //!< Tensor held inside

public:
    /** \brief Creates any_tensor and places a tensor object inside
     **/
    template<typename Tensor>
    any_tensor(Tensor &t);

    /** \brief Copy constructor
     **/
    any_tensor(const any_tensor<N, T> &other) :
        m_tensor(other.m_tensor ? other.m_tensor->clone() : 0)
    { }

    /** \brief Destructor
     **/
    virtual ~any_tensor();

    /** \brief Returns the type of the tensor held inside as a string
     **/
    virtual const char *get_tensor_type() const {
        static const char k_tensor_type[] = "any_tensor";
        return k_tensor_type;
    }

    /** \brief Returns the tensor object held inside as a concrete tensor type
     **/
    template<typename Tensor>
    Tensor &get_tensor() const;

    /** \brief Attaches a letter label to the tensor
     **/
    expr_rhs<N, T> operator()(const label<N> &l);

protected:
    /** \brief Constructor that can only be used by derived classes
     **/
    any_tensor() : m_tensor(0) { }

    /** \brief Actual implementation of operator(), to be redefined in derived
            classes if necessary
     **/
    virtual expr_rhs<N, T> make_rhs(const label<N> &l);

private:
    const any_tensor &operator=(const any_tensor&);

};


template<size_t N, typename T> template<typename Tensor>
any_tensor<N, T>::any_tensor(Tensor &t) : m_tensor(new holder<Tensor>(t)) {

}


template<size_t N, typename T>
any_tensor<N, T>::~any_tensor() {

    delete m_tensor;
}


template<size_t N, typename T> template<typename Tensor>
Tensor &any_tensor<N, T>::get_tensor() const {

    if(m_tensor->type_info() == typeid(Tensor)) {
        return static_cast< holder<Tensor>* >(m_tensor)->m_t;
    } else {
        throw std::bad_cast();
    }
}


template<size_t N, typename T>
expr_rhs<N, T> any_tensor<N, T>::operator()(const label<N> &l) {

    return make_rhs(l);
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::any_tensor;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_ANY_TENSOR_H
