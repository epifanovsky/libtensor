#ifndef LIBTENSOR_IFACE_ANY_TENSOR_H
#define LIBTENSOR_IFACE_ANY_TENSOR_H

#include <typeinfo>

namespace libtensor {
namespace iface {


/** \brief Any tensor type
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This template implements the any concept for tensors. The actual tensor
    type is concealed and is only known to the creator and the recipient.

    \ingroup libtensor_iface
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
        holder(Tensor &t);
        virtual ~holder();
        virtual const std::type_info &type_info() const;
        virtual placeholder *clone() const;
    };

private:
    placeholder *m_tensor; // !< Tensor held inside

public:
    template<typename Tensor>
    Tensor &recast_as();

};


template<size_t N, typename T> template<typename Tensor>
Tensor &any_tensor::recast_as() {

    return dynamic_cast<Tensor&>(m_tensor->m_t);
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_ANY_TENSOR_H
