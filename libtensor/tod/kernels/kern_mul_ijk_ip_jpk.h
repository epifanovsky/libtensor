#ifndef LIBTENSOR_KERN_MUL_IJK_IP_JPK_H
#define LIBTENSOR_KERN_MUL_IJK_IP_JPK_H

#include "kern_mul_ij_ip_pj.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ijk_ip_jpk : public kernel_base<2, 1> {
    friend class kern_mul_ijk_piq_jpqk;
    friend class kern_mul_ijk_piq_pjqk;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_nk, m_np;
    size_t m_sia, m_sjb, m_spb, m_sic, m_sjc;

public:
    virtual ~kern_mul_ijk_ip_jpk() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_mul_ij_ip_pj &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_IP_JPK_H
