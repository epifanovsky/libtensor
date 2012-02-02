#ifndef LIBTENSOR_ANYTENSOR_H
#define LIBTENSOR_ANYTENSOR_H

#include <memory>
#include <typeinfo>
#include "labeled_anytensor.h"

namespace libtensor {


/** \brief Placeholder in tensor expressions
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class anytensor {
private:
//    class placeholder {
//    public:
//        virtual ~placeholder() { }
//        virtual const std::type_info &type_info() const = 0;
//    };
//
//    template<typename TensorType>
//    class holder : public placeholder {
//    public:
//        TensorType &m_held;
//    public:
//        explicit holder(TensorType &t) : m_held(t) { }
//        virtual ~holder() { }
//        virtual const std::type_info &type_info() const {
//            return typeid(TensorType);
//        }
//    };

private:
//    placeholder *m_content;

public:
    /** \brief Default constructor
     **/
    anytensor() //:
        //m_content(0) { }
    {}

    /** \brief Initializes the placeholder by installing a tensor
        \tparam TensorType Type of installed tensor.
        \param t Contained tensor.
     **/
    template<typename TensorType>
    explicit anytensor(TensorType &t) //:
//        m_content(new holder<TensorType>(t)) { }
    {}

    /** \brief Virtual destructor
     **/
    virtual ~anytensor() {
//        delete m_content;
    }

    /** \brief Returns the type of the tensor
     **/
    virtual const char *get_tensor_type() const = 0;

    /** \brief Attaches a letter index label to this anytensor and returns
            itself as a labeled_anytensor
     **/
    labeled_anytensor<N, T> operator()(const letter_expr<N> &label) {
        return labeled_anytensor<N, T>(*this, label);
    }

private:
    anytensor(const anytensor<N, T>&);

};


} // namespace libtensor

#endif // LIBTENSOR_ANYTENSOR_H
