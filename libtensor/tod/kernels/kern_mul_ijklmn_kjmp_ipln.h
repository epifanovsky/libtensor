#ifndef LIBTENSOR_KERN_MUL_IJKLMN_KJMP_IPLN_H
#define LIBTENSOR_KERN_MUL_IJKLMN_KJMP_IPLN_H

#include "kern_mul_ijklm_jlp_ipkm.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ijklmn_kjmp_ipln : public kernel_base<2, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_nk, m_nl, m_nm, m_nn, m_np;
    size_t m_ska, m_sja, m_sma, m_sib, m_spb, m_slb, m_sic, m_sjc, m_skc,
        m_slc, m_smc;

public:
    virtual ~kern_mul_ijklmn_kjmp_ipln() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_mul_ijklm_jlp_ipkm &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJKLM_JLP_IPKM_H
