#ifndef LIBTENSOR_KERN_DADD2_I_I_X_X_H
#define LIBTENSOR_KERN_DADD2_I_I_X_X_H

#include "../kern_dadd2.h"

namespace libtensor {


/** \brief Kernel for \f$ c_i = c_i + (a_i + b) d \f$
    \tparam LA Linear algebra.

     \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dadd2_i_i_x_x : public kernel_base<LA, 2, 1> {
public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 2, 1>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1>::iterator_t iterator_t;

private:
    double m_ka, m_kb;
    double m_d;
    size_t m_ni;
    size_t m_sia, m_sic;

public:
    virtual ~kern_dadd2_i_i_x_x() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers<2, 1> &r);

    static kernel_base<LA, 2, 1> *match(const kern_dadd2<LA> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DADD2_I_I_X_X_H
