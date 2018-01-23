#ifndef LIBTENSOR_KERN_ADD2_H
#define LIBTENSOR_KERN_ADD2_H

#include "kernel_base.h"

namespace libtensor {


template<typename LA, typename T> class kern_add2_i_i_x_x;
template<typename LA, typename T> class kern_add2_i_x_i_x;


/** \brief Generic addition kernel (T)
    \tparam LA Linear algebra.

    This kernel adds two multidimensional arrays with optional scaling:
    \f[
        c = c + d (k_a a + k_b b)
    \f]
    a, b, c are arrays, d, k_a, k_b are scaling factors.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_add2 : public kernel_base<LA, 2, 1, T> {
    friend class kern_add2_i_i_x_x<LA, T>;
    friend class kern_add2_i_x_i_x<LA, T>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1, T>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 2, 1, T>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1, T>::iterator_t iterator_t;

private:
    T m_ka, m_kb;
    T m_d;

public:
    virtual ~kern_add2() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<2, 1, T> &r);

    static kernel_base<LA, 2, 1, T> *match(T ka, T kb, T d,
        list_t &in, list_t &out);

};

template<typename LA>
using kern_dadd2 = kern_add2<LA, double>; 

} // namespace libtensor

#endif // LIBTENSOR_KERN_ADD2_H
