#ifndef LIBTENSOR_KERN_ADD1_H
#define LIBTENSOR_KERN_ADD1_H

#include "kernel_base.h"

namespace libtensor {


template<typename LA, typename T> class kern_add1_i_i_x;


/** \brief Generic addition-to kernel (double and float)
    \tparam LA Linear algebra.

    This kernel adds to a multidimensional array with optional scaling:
    \f[
        b = b + d a
    \f]
    a, b are arrays, d is a scaling factor.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_add1 : public kernel_base<LA, 1, 1, T> {
    friend class kern_add1_i_i_x<LA, T>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 1, 1, T>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 1, 1, T>::list_t list_t;
    typedef typename kernel_base<LA, 1, 1, T>::iterator_t iterator_t;

private:
    T m_d;

public:
    virtual ~kern_add1() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<1, 1, T> &r);

    static kernel_base<LA, 1, 1, T> *match(T d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_ADD1_H
