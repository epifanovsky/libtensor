#ifndef LIBTENSOR_KERN_MUL_IJ_PJI_P_H
#define LIBTENSOR_KERN_MUL_IJ_PJI_P_H

#include "kern_mul_i_pi_p.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ij_pji_p : public kernel_base<2, 1> {
    friend class kern_mul_ijk_pkj_ip;
    friend class kern_mul_ijk_pkj_pi;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_np;
    size_t m_sja, m_spa, m_spb, m_sic;

public:
    virtual ~kern_mul_ij_pji_p() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2_i_pi_p &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJ_PJI_P_H
