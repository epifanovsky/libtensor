#ifndef LIBTENSOR_KERN_DDIVADD1_H
#define LIBTENSOR_KERN_DDIVADD1_H

#include <libtensor/linalg/linalg.h>
#include "kernel_base.h"

namespace libtensor {


/** \brief Generic elementwise division with addition kernel (double)

    This kernel performs the division-addition of a multidimensional array
    elementwise with optional scaling:
    \f[
        b = b + d \frac{b}{a}
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
class kern_ddivadd1 : public kernel_base<linalg, 1, 1, double> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;

public:
    virtual ~kern_ddivadd1() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(void*, const loop_registers<1, 1> &r);

    static kernel_base<linalg, 1, 1, double> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DDIVADD1_H
