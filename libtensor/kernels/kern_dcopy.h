#ifndef LIBTENSOR_KERN_DCOPY_H
#define LIBTENSOR_KERN_DCOPY_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Generic copy kernel (double)

    This kernel copies a multidimensional array with optional scaling:
    \f[
        b = d a
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
class kern_dcopy : public kernel_base<1, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;

public:
    virtual ~kern_dcopy() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<1, 1> &r);

    static kernel_base<1, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DCOPY_H
