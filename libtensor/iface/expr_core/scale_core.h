#ifndef LIBTENSOR_IFACE_SCALE_CORE_H
#define LIBTENSOR_IFACE_SCALE_CORE_H

#include "../expr_core_i.h"

namespace libtensor {
namespace iface {


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
    expr_core_ptr<N, T> m_expr; //!< Unscaled expression

public:
    /** \brief Constructs the scaling expression using a coefficient
            and the underlying unscaled expression
     **/
    scale_core(const T &coeff, const expr_core_ptr<N, T> &subexpr) :
        m_coeff(coeff), m_expr(subexpr)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~scale_core() { }

    /** \brief Returns the unscaled expression
     **/
    expr_core_ptr<N, T> &get_unscaled_expr() {
        return m_expr;
    }

    /** \brief Returns the unscaled expression (const version)
     **/
    const expr_core_ptr<N, T> &get_unscaled_expr() const {
        return m_expr;
    }

    /** \brief Returns the scaling coefficient
     **/
    const T &get_coeff() {
        return m_coeff;
    }
};


template<size_t N, typename T>
const char scale_core<N, T>::k_clazz = "scale_core<N, T>";


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_SCALE_CORE_H
