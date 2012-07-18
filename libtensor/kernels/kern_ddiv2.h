#ifndef LIBTENSOR_KERN_DDIV2_H
#define LIBTENSOR_KERN_DDIV2_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Generic division kernel (double)

    This kernel divides two multidimensional arrays with optional scaling:
    \f[
        c = c + d \frac{a}{b}
    \f]
    a, b, c are arrays, d is a scaling factors.

    \ingroup libtensor_kernels
 **/
class kern_ddiv2 : public kernel_base<2, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;

public:
    virtual ~kern_ddiv2() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DDIV2_H
