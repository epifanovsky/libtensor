#ifndef LIBTENSOR_KERN_MUL2_X_P_P_H
#define LIBTENSOR_KERN_MUL2_X_P_P_H

#include "../kern_mul2.h"

namespace libtensor {


template<typename LA, typename T> class kern_mul2_i_ip_p;
template<typename LA, typename T> class kern_mul2_i_p_ip;
template<typename LA, typename T> class kern_mul2_x_pq_pq;
template<typename LA, typename T> class kern_mul2_x_pq_qp;


/** \brief Specialized kernel for \f$ c = c + a_p b_p \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA, typename T>
class kern_mul2_x_p_p : public kernel_base<LA, 2, 1, T> {
    friend class kern_mul2_i_ip_p<LA, T>;
    friend class kern_mul2_i_p_ip<LA, T>;
    friend class kern_mul2_x_pq_pq<LA, T>;
    friend class kern_mul2_x_pq_qp<LA, T>;

public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1, T>::device_context_ref
            device_context_ref;
    typedef typename kernel_base<LA, 2, 1, T>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1, T>::iterator_t iterator_t;

private:
    T m_d;
    size_t m_np;
    size_t m_spa, m_spb;

public:
    virtual ~kern_mul2_x_p_p() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers_x<2, 1, T> &r);

    static kernel_base<LA, 2, 1, T> *match(const kern_mul2<LA, T> &z,
            list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_X_P_P_H
