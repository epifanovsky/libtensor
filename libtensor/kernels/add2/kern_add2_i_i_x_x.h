#ifndef LIBTENSOR_KERN_ADD2_I_I_X_X_H
#define LIBTENSOR_KERN_ADD2_I_I_X_X_H

#include "../kern_add2.h"

namespace libtensor {


/** \brief Kernel for \f$ c_i = c_i + (a_i + b) d \f$
    \tparam LA Linear algebra.

     \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_add2_i_i_x_x : public kernel_base<LA, 2, 1, T> {
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
    size_t m_ni;
    size_t m_sia, m_sic;

public:
    virtual ~kern_add2_i_i_x_x() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<2, 1, T> &r);

    static kernel_base<LA, 2, 1, T> *match(const kern_add2<LA, T> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_ADD2_I_I_X_X_H
