#ifndef LIBTENSOR_KERN_MUL_IJK_PJ_IPK_H
#define LIBTENSOR_KERN_MUL_IJK_PJ_IPK_H

#include "kern_mul_ij_pi_pj.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ijk_pj_ipk : public kernel_base<2, 1> {
    friend class kern_mul_ijk_pqj_iqpk;
    friend class kern_mul_ijk_pqj_qipk;
    friend class kern_mul_ijkl_ipk_jpl;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_nk, m_np;
    size_t m_spa, m_sib, m_spb, m_sic, m_sjc;

public:
    virtual ~kern_mul_ijk_pj_ipk() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2_ij_pi_pj &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_PJ_IPK_H
