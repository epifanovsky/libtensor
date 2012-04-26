#ifndef LIBTENSOR_KERN_MUL_IJK_PKQ_PIQJ_H
#define LIBTENSOR_KERN_MUL_IJK_PKQ_PIQJ_H

#include "kern_mul_ijk_kp_ipj.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ijk_pkq_piqj : public kernel_base<2, 1> {
    friend class kern_mul_ijkl_pliq_pjqk;
    friend class kern_mul_ijkl_pljq_piqk;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_nk, m_np, m_nq;
    size_t m_spa, m_ska, m_spb, m_sib, m_sqb, m_sic, m_sjc;

public:
    virtual ~kern_mul_ijk_pkq_piqj() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_mul_ijk_kp_ipj &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_PKQ_PIQJ_H
