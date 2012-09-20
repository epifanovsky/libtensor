#ifndef LIBTENSOR_KERN_DMUL2_I_IP_P_H
#define LIBTENSOR_KERN_DMUL2_I_IP_P_H

#include "kern_dmul2_x_p_p.h"

namespace libtensor {


template<typename LA> class kern_dmul2_ij_jp_ip;


/** \brief Specialized kernel for \f$ c_i = c_i + a_{ip} b_p d \f$
    \tparam LA Linear algebra.

     \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dmul2_i_ip_p : public kernel_base<LA, 2, 1> {
    friend class kern_dmul2_ij_jp_ip<LA>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 2, 1>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1>::iterator_t iterator_t;

private:
    double m_d;
    size_t m_ni, m_np;
    size_t m_sia, m_spb, m_sic;

public:
    virtual ~kern_dmul2_i_ip_p() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers<2, 1> &r);

    static kernel_base<LA, 2, 1> *match(const kern_dmul2_x_p_p<LA> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_I_IP_P_H
