#ifndef LIBTENSOR_KERN_DADD2_H
#define LIBTENSOR_KERN_DADD2_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Generic addition kernel (double)

    This kernel adds two multidimensional arrays with optional scaling:
    \f[
        c = c + d (k_a a + k_b b)
    \f]
    a, b, c are arrays, d, k_a, k_b are scaling factors.

    \ingroup libtensor_kernels
 **/
class kern_dadd2 : public kernel_base<2, 1> {
    friend class kern_dadd2_i_i_x_x;
    friend class kern_dadd2_i_x_i_x;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_ka, m_kb;
    double m_d;

public:
    virtual ~kern_dadd2() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(double ka, double kb, double d, list_t &in,
        list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DADD2_H
