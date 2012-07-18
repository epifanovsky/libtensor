#ifndef LIBTENSOR_KERN_MUL_IJ_P_PJI_H
#define LIBTENSOR_KERN_MUL_IJ_P_PJI_H

#include "kern_mul_i_p_pi.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ij_p_pji : public kernel_base<2, 1> {
    friend class kern_mul_ijk_ip_pkj;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_np;
    size_t m_spa, m_sjb, m_spb, m_sic;

public:
    virtual ~kern_mul_ij_p_pji() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2_i_p_pi &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJ_P_PJI_H
