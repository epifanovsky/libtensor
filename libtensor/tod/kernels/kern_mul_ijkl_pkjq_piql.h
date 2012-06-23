#ifndef LIBTENSOR_KERN_MUL_IJKL_PKJQ_PIQL_H
#define LIBTENSOR_KERN_MUL_IJKL_PKJQ_PIQL_H

#include "kern_mul_ijk_pjq_piqk.h"

namespace libtensor {


/**
    \ingroup libtensor_tod_kernel
 **/
class kern_mul_ijkl_pkjq_piql : public kernel_base<2, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_nk, m_nl, m_np, m_nq;
    size_t m_spa, m_ska, m_sja, m_spb, m_sib, m_sqb, m_sic, m_sjc, m_skc;

public:
    virtual ~kern_mul_ijkl_pkjq_piql() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_mul_ijk_pjq_piqk &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJKL_PKJQ_PIQL_H
