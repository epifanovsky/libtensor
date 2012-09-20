#ifndef LIBTENSOR_KERN_DADD1_I_I_X_H
#define LIBTENSOR_KERN_DADD1_I_I_X_H

#include "../kern_dadd1.h"

namespace libtensor {


template<typename LA> class kern_dadd1_ij_ij_x;
template<typename LA> class kern_dadd1_ij_ji_x;


/** \brief Specialized kernel for \f$ b_i = b_i + a_i d \f$
    \tparam LA Linear algebra.

     \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dadd1_i_i_x : public kernel_base<LA, 1, 1> {
    friend class kern_dadd1_ij_ij_x<LA>;
    friend class kern_dadd1_ij_ji_x<LA>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 1, 1>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 1, 1>::list_t list_t;
    typedef typename kernel_base<LA, 1, 1>::iterator_t iterator_t;

private:
    double m_d;
    size_t m_ni;
    size_t m_sia, m_sib;

public:
    virtual ~kern_dadd1_i_i_x() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers<1, 1> &r);

    static kernel_base<LA, 1, 1> *match(const kern_dadd1<LA> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DADD1_I_I_X_H
