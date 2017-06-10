#ifndef LIBTENSOR_KERN_DMUL2_I_P_PI_H
#define LIBTENSOR_KERN_DMUL2_I_P_PI_H

#include "kern_dmul2_i_x_i.h"

namespace libtensor {


template<typename LA> class kern_dmul2_ij_ip_pj;
template<typename LA> class kern_dmul2_ij_jp_pi;
template<typename LA> class kern_dmul2_ij_pi_pj;


/** \brief Specialized kernel for \f$ c_i = c_i + a_p b_{pi} d \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dmul2_i_p_pi : public kernel_base<LA, 2, 1, double> {
    friend class kern_dmul2_ij_ip_pj<LA>;
    friend class kern_dmul2_ij_jp_pi<LA>;
    friend class kern_dmul2_ij_pi_pj<LA>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1, double>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 2, 1, double>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1, double>::iterator_t iterator_t;

private:
    double m_d;
    size_t m_ni, m_np;
    size_t m_spa, m_spb, m_sic;

public:
    virtual ~kern_dmul2_i_p_pi() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers<2, 1> &r);

    static kernel_base<LA, 2, 1, double> *match(const kern_dmul2_i_x_i<LA> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_I_P_PI_H
