#ifndef LIBTENSOR_IFACE_IDENT_CORE_H
#define LIBTENSOR_IFACE_IDENT_CORE_H

#include "../expr_core_i.h"
#include "../any_tensor.h"

namespace libtensor {
namespace iface {


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

public:
    /** \brief Initializes the operation with a tensor reference
     **/
    ident_core(const any_tensor<N, T> &t) : m_t(t)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~ident_core() { }

    /** \brief Returns type of the expression
     **/
    virtual const std::string &get_type() const {
        return "ident";
    }

    /** \brief Returns the enclosed tensor
     **/
    const any_tensor<N, T> &get_tensor() const {
        return m_t;
    }
};


template<size_t N, typename T>
const char ident_core<N, T>::k_clazz[] = "ident_core<N, T>";



} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_IDENT_CORE_H
