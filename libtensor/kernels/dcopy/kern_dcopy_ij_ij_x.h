#ifndef LIBTENSOR_KERN_DCOPY_IJ_IJ_X_H
#define LIBTENSOR_KERN_DCOPY_IJ_IJ_X_H

#include "kern_dcopy_i_i_x.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ b_{ij} = a_{ij} d \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dcopy_ij_ij_x : public kernel_base<LA, 1, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni, m_nj;
    size_t m_sia, m_sib;

public:
    typedef typename kernel_base<LA, 1, 1>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 1, 1>::list_t list_t;
    typedef typename kernel_base<LA, 1, 1>::iterator_t iterator_t;

public:
    virtual ~kern_dcopy_ij_ij_x() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers<1, 1> &r);

    static kernel_base<LA, 1, 1> *match(const kern_dcopy_i_i_x<LA> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DCOPY_IJ_IJ_X_H
