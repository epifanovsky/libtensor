#ifndef LIBTENSOR_KERN_DDIV1_H
#define LIBTENSOR_KERN_DDIV1_H

#include "kernel_base.h"

namespace libtensor {


template<typename LA> class kern_ddiv1_i_i_x;


/** \brief Generic elementwise division kernel (double)

    This kernel performs the division of a multidimensional array elementwise
    with optional scaling:
    \f[
        b = d \frac{b}{a}
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_ddiv1 : public kernel_base<LA, 1, 1, double> {
    friend class kern_ddiv1_i_i_x<LA>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 1, 1, double>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 1, 1, double>::list_t list_t;
    typedef typename kernel_base<LA, 1, 1, double>::iterator_t iterator_t;

private:
    double m_d;

public:
    virtual ~kern_ddiv1() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers<1, 1> &r);

    static kernel_base<LA, 1, 1, double> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DDIV1_H
