#ifndef LIBTENSOR_KERN_DADD1_H
#define LIBTENSOR_KERN_DADD1_H

#include "kernel_base.h"

namespace libtensor {


template<typename LA> class kern_dadd1_i_i_x;


/** \brief Generic addition-to kernel (double)
    \tparam LA Linear algebra.

    This kernel adds to a multidimensional array with optional scaling:
    \f[
        b = b + d a
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dadd1 : public kernel_base<LA, 1, 1> {
    friend class kern_dadd1_i_i_x<LA>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 1, 1>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 1, 1>::list_t list_t;
    typedef typename kernel_base<LA, 1, 1>::iterator_t iterator_t;

private:
    double m_d;

public:
    virtual ~kern_dadd1() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers<1, 1> &r);

    static kernel_base<LA, 1, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DADD1_H
