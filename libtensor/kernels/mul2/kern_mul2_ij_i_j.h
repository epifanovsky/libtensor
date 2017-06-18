#ifndef LIBTENSOR_KERN_MUL2_IJ_I_J_H
#define LIBTENSOR_KERN_MUL2_IJ_I_J_H

#include "kern_mul2_i_i_x.h"

namespace libtensor {


template<typename LA, typename T> class kern_mul2_ij_ip_pj;


/** \brief Specialized kernel for \f$ c_{ij} = c_{ij} + a_i b_j d \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_mul2_ij_i_j : public kernel_base<LA, 2, 1, T> {
    friend class kern_mul2_ij_ip_pj<LA, T>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1, T>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 2, 1, T>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1, T>::iterator_t iterator_t;

private:
    T m_d;
    size_t m_ni, m_nj;
    size_t m_sia, m_sjb, m_sic;

public:
    virtual ~kern_mul2_ij_i_j() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<2, 1, T> &r);

    static kernel_base<LA, 2, 1, T> *match(const kern_mul2_i_i_x<LA, T> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_IJ_I_J_H