#ifndef LIBTENSOR_KERN_DMUL2_IJ_PJ_IP_H
#define LIBTENSOR_KERN_DMUL2_IJ_PJ_IP_H

#include "kern_dmul2_i_pi_p.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ c_{ij} = c_{ij} + a_{pj} b_{ip} d \f$
    \tparam LA Linear algebra.

     \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dmul2_ij_pj_ip : public kernel_base<2, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj, m_np;
    size_t m_spa, m_sib, m_sic;

public:
    virtual ~kern_dmul2_ij_pj_ip() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2_i_pi_p<LA> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_IJ_PJ_IP_H
