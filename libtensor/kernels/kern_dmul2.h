#ifndef LIBTENSOR_KERN_DMUL2_H
#define LIBTENSOR_KERN_DMUL2_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Generic multiplication kernel (double)

    This kernel multiplies two multidimensional arrays with optional scaling:
    \f[
        c = c + d a b
    \f]
    a, b, c are arrays, d is a scaling factors.

    \ingroup libtensor_kernels
 **/
class kern_dmul2 : public kernel_base<2, 1> {
    friend class kern_dmul2_i_i_i;
    friend class kern_dmul2_i_i_x;
    friend class kern_dmul2_i_x_i;
    friend class kern_dmul2_x_p_p;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;

public:
    virtual ~kern_dmul2() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_H
