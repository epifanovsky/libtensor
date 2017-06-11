#ifndef LIBTENSOR_KERN_MUL2_IJ_PI_JP_H
#define LIBTENSOR_KERN_MUL2_IJ_PI_JP_H

#include "kern_mul2_i_pi_p.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ c_{ij} = c_{ij} + a_{pi} b_{jp} d \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_mul2_ij_pi_jp : public kernel_base<LA, 2, 1, T> {
public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1, T>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 2, 1, T>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1, T>::iterator_t iterator_t;

private:
    T m_d;
    size_t m_ni, m_nj, m_np;
    size_t m_spa, m_sjb, m_sic;

public:
    virtual ~kern_mul2_ij_pi_jp() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<2, 1, T> &r);

    static kernel_base<LA, 2, 1, T> *match(const kern_mul2_i_pi_p<LA, T> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_IJ_PI_JP_H
