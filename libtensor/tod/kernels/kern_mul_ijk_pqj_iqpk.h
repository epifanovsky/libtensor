#ifndef LIBTENSOR_KERN_MUL_IJK_PQJ_IQPK_H
#define LIBTENSOR_KERN_MUL_IJK_PQJ_IQPK_H

#include "kern_mul_ijk_pj_ipk.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ijk_pqj_iqpk : public kernel_base<2, 1> {
    friend class kern_mul_ijkl_pqkj_iqpl;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_nk, m_np, m_nq;
    size_t m_spa, m_sqa, m_sib, m_sqb, m_spb, m_sic, m_sjc;

public:
    virtual ~kern_mul_ijk_pqj_iqpk() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_mul_ijk_pj_ipk &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_PQJ_IQPK_H
