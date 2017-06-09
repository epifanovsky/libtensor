#ifndef LIBTENSOR_KERN_COPY_I_I_X_H
#define LIBTENSOR_KERN_COPY_I_I_X_H

#include "../kern_copy.h"

namespace libtensor {


template<typename LA, typename T> class kern_copy_ij_ij_x;
template<typename LA, typename T> class kern_copy_ij_ji_x;


/** \brief Specialized kernel for \f$ b_i = a_i d \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_copy_i_i_x : public kernel_base<LA, 1, 1> {
    friend class kern_copy_ij_ij_x<LA, T>;
    friend class kern_copy_ij_ji_x<LA, T>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 1, 1>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 1, 1>::list_t list_t;
    typedef typename kernel_base<LA, 1, 1>::iterator_t iterator_t;

private:
    T m_d;
    size_t m_ni;
    size_t m_sia, m_sib;

public:
    virtual ~kern_copy_i_i_x() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<1, 1, T> &r);

    static kernel_base<LA, 1, 1> *match(const kern_copy<LA, T> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_COPY_I_I_X_H
