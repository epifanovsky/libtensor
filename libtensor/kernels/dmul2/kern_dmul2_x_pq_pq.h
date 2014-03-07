#ifndef LIBTENSOR_KERN_DMUL2_X_PQ_PQ_H
#define LIBTENSOR_KERN_DMUL2_X_PQ_PQ_H

#include "../kern_dmul2.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ c = c + a_pq b_pq \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dmul2_x_pq_pq : public kernel_base<LA, 2, 1> {
public:
    static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1>::device_context_ref
            device_context_ref;
    typedef typename kernel_base<LA, 2, 1>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1>::iterator_t iterator_t;

private:
    double m_d;
    size_t m_np, m_nq;
    size_t m_spa, m_spb;

public:
    virtual ~kern_dmul2_x_pq_pq() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(device_context_ref ctx, const loop_registers<2, 1> &r);

    static kernel_base<LA, 2, 1> *match(const kern_dmul2_x_p_p<LA> &z,
            list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_X_P_P_H
