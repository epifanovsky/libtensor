#ifndef LIBTENSOR_KERN_DMUL2_I_PI_P_H
#define LIBTENSOR_KERN_DMUL2_I_PI_P_H

#include "kern_dmul2_i_i_x.h"

namespace libtensor {


template<typename LA> class kern_dmul2_ij_pi_jp;
template<typename LA> class kern_dmul2_ij_pi_pj;
template<typename LA> class kern_dmul2_ij_pj_ip;
template<typename LA> class kern_dmul2_ij_pj_pi;


/** \brief Specialized kernel for \f$ c_i = c_i + a_{pi} b_p d \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dmul2_i_pi_p : public kernel_base<2, 1> {
    friend class kern_dmul2_ij_pi_jp<LA>;
    friend class kern_dmul2_ij_pi_pj<LA>;
    friend class kern_dmul2_ij_pj_ip<LA>;
    friend class kern_dmul2_ij_pj_pi<LA>;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_np;
    size_t m_spa, m_spb, m_sic;

public:
    virtual ~kern_dmul2_i_pi_p() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2_i_i_x<LA> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_I_PI_P_H
