#ifndef LIBTENSOR_KERN_MUL_IJKL_JKP_IPL_H
#define LIBTENSOR_KERN_MUL_IJKL_JKP_IPL_H

#include "kern_mul_ijk_jp_ipk.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ijkl_jkp_ipl : public kernel_base<2, 1> {
    friend class kern_mul_ijklm_jlp_ipkm;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_nk, m_nl, m_np;
    size_t m_sja, m_ska, m_sib, m_spb, m_sic, m_sjc, m_skc;

public:
    virtual ~kern_mul_ijkl_jkp_ipl() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_mul_ijk_jp_ipk &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJKL_JKP_IPL_H
