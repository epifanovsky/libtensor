#ifndef LIBTENSOR_KERN_DMULADD1_H
#define LIBTENSOR_KERN_DMULADD1_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Generic elementwise multiplication with addition kernel (double)

    This kernel performs multiply-add on a multidimensional array elementwise
    with optional scaling:
    \f[
        b = b + d a b
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
class kern_dmuladd1 : public kernel_base<1, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;

public:
    virtual ~kern_dmuladd1() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<1, 1> &r);

    static kernel_base<1, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMULADD1_H
