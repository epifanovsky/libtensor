#ifndef LIBTENSOR_KERN_DMUL2_IJ_I_J_H
#define LIBTENSOR_KERN_DMUL2_IJ_I_J_H

#include "kern_dmul2_i_i_x.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ c_{ij} = c_{ij} + a_i b_j d \f$

     \ingroup libtensor_kernels
 **/
class kern_dmul2_ij_i_j : public kernel_base<2, 1> {
    friend class kern_dmul2_ij_ip_pj;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj;
    size_t m_sia, m_sjb, m_sic;

public:
    virtual ~kern_dmul2_ij_i_j() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2_i_i_x &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_IJ_I_J_H
