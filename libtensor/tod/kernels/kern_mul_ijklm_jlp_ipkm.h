#ifndef LIBTENSOR_KERN_MUL_IJKLM_JLP_IPKM_H
#define LIBTENSOR_KERN_MUL_IJKLM_JLP_IPKM_H

#include "kern_mul_ijkl_jkp_ipl.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ijklm_jlp_ipkm : public kernel_base<2, 1> {
    friend class kern_mul_ijklmn_kjmp_ipln;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_nk, m_nl, m_nm, m_np;
    size_t m_sja, m_sla, m_sib, m_spb, m_skb, m_sic, m_sjc, m_skc, m_slc;

public:
    virtual ~kern_mul_ijklm_jlp_ipkm() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_mul_ijkl_jkp_ipl &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJKLM_JLP_IPKM_H
