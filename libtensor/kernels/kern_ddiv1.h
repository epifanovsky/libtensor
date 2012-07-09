#ifndef LIBTENSOR_KERN_DDIV1_H
#define LIBTENSOR_KERN_DDIV1_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Generic elementwise division kernel (double)

    This kernel performs the division of a multidimensional array elementwise
    with optional scaling:
    \f[
        b = d \frac{b}{a}
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
class kern_ddiv1 : public kernel_base<1, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;

public:
    virtual ~kern_ddiv1() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<1, 1> &r);

    static kernel_base<1, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DDIV1_H
