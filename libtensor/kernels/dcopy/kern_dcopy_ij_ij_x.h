#ifndef LIBTENSOR_KERN_DCOPY_IJ_IJ_X_H
#define LIBTENSOR_KERN_DCOPY_IJ_IJ_X_H

#include "kern_dcopy_i_i_x.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ b_{ij} = a_{ij} d \f$

    \ingroup libtensor_kernels
 **/
class kern_dcopy_ij_ij_x : public kernel_base<1, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj;
    size_t m_sia, m_sib;

public:
    virtual ~kern_dcopy_ij_ij_x() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<1, 1> &r);

    static kernel_base<1, 1> *match(const kern_dcopy_i_i_x &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DCOPY_IJ_IJ_X_H
