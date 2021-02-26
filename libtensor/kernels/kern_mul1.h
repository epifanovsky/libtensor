#ifndef LIBTENSOR_KERN_MUL1_H
#define LIBTENSOR_KERN_MUL1_H

#include <libtensor/linalg/linalg.h>
#include "kernel_base.h"

namespace libtensor {


/** \brief Generic elementwise multiplication kernel (T)

    This kernel multiplies a multidimensional array elementwise with optional
    scaling:
    \f[
        b = d a b
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
template<typename T>
class kern_mul1 : public kernel_base<linalg, 1, 1, T> {
public:
    typedef std::list< loop_list_node<1, 1> > list_t;
    static const char *k_clazz; //!< Kernel name

private:
    T m_d;

public:
    virtual ~kern_mul1() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(void*, const loop_registers_x<1, 1, T> &r);

    static kernel_base<linalg, 1, 1, T> *match(T d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL1_H
