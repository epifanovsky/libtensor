#ifndef LIBTENSOR_KERN_MUL2_H
#define LIBTENSOR_KERN_MUL2_H

#include "kernel_base.h"

namespace libtensor {


template<typename LA, typename T> class kern_mul2_i_i_i;
template<typename LA, typename T> class kern_mul2_i_i_x;
template<typename LA, typename T> class kern_mul2_i_x_i;
template<typename LA, typename T> class kern_mul2_x_p_p;


/** \brief Generic multiplication kernel (T)
    \tparam LA Linear algebra.

    This kernel multiplies two multidimensional arrays with optional scaling:
    \f[
        c = c + d a b
    \f]
    a, b, c are arrays, d is a scaling factors.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_mul2 : public kernel_base<LA, 2, 1, T> {
    friend class kern_mul2_i_i_i<LA, T>;
    friend class kern_mul2_i_i_x<LA, T>;
    friend class kern_mul2_i_x_i<LA, T>;
    friend class kern_mul2_x_p_p<LA, T>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1, T>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 2, 1, T>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1, T>::iterator_t iterator_t;

private:
    T m_d;

public:
    virtual ~kern_mul2() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<2, 1, T> &r);

    static kernel_base<LA, 2, 1, T> *match(T d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_H
