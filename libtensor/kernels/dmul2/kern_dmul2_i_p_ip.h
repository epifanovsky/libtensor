#ifndef LIBTENSOR_KERN_DMUL2_I_P_IP_H
#define LIBTENSOR_KERN_DMUL2_I_P_IP_H

#include "kern_dmul2_x_p_p.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ c_i = c_i + a_p b_{ip} d \f$

    \ingroup libtensor_kernels
 **/
class kern_dmul2_i_p_ip : public kernel_base<2, 1> {
    friend class kern_dmul2_ij_ip_jp;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_np;
    size_t m_spa, m_sib, m_sic;

public:
    virtual ~kern_dmul2_i_p_ip() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2_x_p_p &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_I_P_IP_H
