#ifndef LIBTENSOR_KERN_DIV1_H
#define LIBTENSOR_KERN_DIV1_H

#include "kernel_base.h"

namespace libtensor {


template<typename LA, typename T> class kern_div1_i_i_x;


/** \brief Generic elementwise division kernel (T)

    This kernel performs the division of a multidimensional array elementwise
    with optional scaling:
    \f[
        b = d \frac{b}{a}
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_div1 : public kernel_base<LA, 1, 1, T> {
    friend class kern_div1_i_i_x<LA, T>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 1, 1, T>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 1, 1, T>::list_t list_t;
    typedef typename kernel_base<LA, 1, 1, T>::iterator_t iterator_t;

private:
    T m_d;

public:
    virtual ~kern_div1() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<1, 1, T> &r);

    static kernel_base<LA, 1, 1, T> *match(T d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DIV1_H
