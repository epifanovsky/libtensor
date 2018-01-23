#ifndef LIBTENSOR_KERN_MUL2_I_P_PI_H
#define LIBTENSOR_KERN_MUL2_I_P_PI_H

#include "kern_mul2_i_x_i.h"

namespace libtensor {


template<typename LA, typename T> class kern_mul2_ij_ip_pj;
template<typename LA, typename T> class kern_mul2_ij_jp_pi;
template<typename LA, typename T> class kern_mul2_ij_pi_pj;


/** \brief Specialized kernel for \f$ c_i = c_i + a_p b_{pi} d \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_mul2_i_p_pi : public kernel_base<LA, 2, 1, T> {
    friend class kern_mul2_ij_ip_pj<LA, T>;
    friend class kern_mul2_ij_jp_pi<LA, T>;
    friend class kern_mul2_ij_pi_pj<LA, T>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1, T>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 2, 1, T>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1, T>::iterator_t iterator_t;

private:
    T m_d;
    size_t m_ni, m_np;
    size_t m_spa, m_spb, m_sic;

public:
    virtual ~kern_mul2_i_p_pi() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<2, 1, T> &r);

    static kernel_base<LA, 2, 1, T> *match(const kern_mul2_i_x_i<LA, T> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_I_P_PI_H
