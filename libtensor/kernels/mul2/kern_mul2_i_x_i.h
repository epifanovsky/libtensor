#ifndef LIBTENSOR_KERN_MUL2_I_X_I_H
#define LIBTENSOR_KERN_MUL2_I_X_I_H

#include "../kern_mul2.h"

namespace libtensor {


template<typename LA, typename T> class kern_mul2_i_p_pi;


/** \brief Specialized kernel for \f$ c_i = c_i + a b_i \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_mul2_i_x_i : public kernel_base<LA, 2, 1, T> {
    friend class kern_mul2_i_p_pi<LA, T>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1, T>::device_context_ref
            device_context_ref;
    typedef typename kernel_base<LA, 2, 1, T>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1, T>::iterator_t iterator_t;

private:
    T m_d;
    size_t m_ni;
    size_t m_sib, m_sic;

public:
    virtual ~kern_mul2_i_x_i() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<2, 1, T> &r);

    static kernel_base<LA, 2, 1, T> *match(const kern_mul2<LA, T> &z,
            list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_I_X_I_H
