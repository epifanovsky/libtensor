#ifndef LIBTENSOR_KERN_MUL_IJK_PIK_PJ_H
#define LIBTENSOR_KERN_MUL_IJK_PIK_PJ_H

#include "kern_mul_ij_pj_pi.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ijk_pik_pj : public kernel_base<2, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_nk, m_np;
    size_t m_spa, m_sia, m_spb, m_sic, m_sjc;

public:
    virtual ~kern_mul_ijk_pik_pj() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2_ij_pj_pi &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_PIK_PJ_H
