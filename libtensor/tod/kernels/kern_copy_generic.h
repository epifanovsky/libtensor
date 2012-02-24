#ifndef LIBTENSOR_KERN_COPY_GENERIC_H
#define LIBTENSOR_KERN_COPY_GENERIC_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Generic kernel for copy

    \ingroup libtensor_tod_kernel
 **/
class kern_copy_generic : public kernel_base<1, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;

public:
    virtual ~kern_copy_generic() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<1, 1> &r);

    static kernel_base<1, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_COPY_GENERIC_H
