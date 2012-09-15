#ifndef LIBTENSOR_KERN_DMUL2_H
#define LIBTENSOR_KERN_DMUL2_H

#include "kernel_base.h"

namespace libtensor {


template<typename LA> class kern_dmul2_i_i_i;
template<typename LA> class kern_dmul2_i_i_x;
template<typename LA> class kern_dmul2_i_x_i;
template<typename LA> class kern_dmul2_x_p_p;


/** \brief Generic multiplication kernel (double)
    \tparam LA Linear algebra.

    This kernel multiplies two multidimensional arrays with optional scaling:
    \f[
        c = c + d a b
    \f]
    a, b, c are arrays, d is a scaling factors.

    \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dmul2 : public kernel_base<2, 1> {
    friend class kern_dmul2_i_i_i<LA>;
    friend class kern_dmul2_i_i_x<LA>;
    friend class kern_dmul2_i_x_i<LA>;
    friend class kern_dmul2_x_p_p<LA>;

public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;

public:
    virtual ~kern_dmul2() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_H
