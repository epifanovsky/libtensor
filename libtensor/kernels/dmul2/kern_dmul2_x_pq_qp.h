#ifndef LIBTENSOR_KERN_DMUL2_X_PQ_QP_H
#define LIBTENSOR_KERN_DMUL2_X_PQ_QP_H

#include "kern_dmul2_x_p_p.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ c = c + a_{pq} b_{qp} \f$

    \ingroup libtensor_kernels
 **/
class kern_dmul2_x_pq_qp : public kernel_base<2, 1> {
    friend class kern_mul_i_ipq_qp;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_np, m_nq;
    size_t m_spa, m_sqb;

public:
    virtual ~kern_dmul2_x_pq_qp() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2_x_p_p &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_X_PQ_QP_H
