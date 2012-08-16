#ifndef LIBTENSOR_KERN_DADD1_H
#define LIBTENSOR_KERN_DADD1_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Generic addition-to kernel (double)

    This kernel adds to a multidimensional array with optional scaling:
    \f[
        b = b + d a
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
class kern_dadd1 : public kernel_base<1, 1> {
    friend class kern_dadd1_i_i_x;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;

public:
    virtual ~kern_dadd1() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<1, 1> &r);

    static kernel_base<1, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DADD1_H
