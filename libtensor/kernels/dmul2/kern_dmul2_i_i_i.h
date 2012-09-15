#ifndef LIBTENSOR_KERN_DMUL2_I_I_I_H
#define LIBTENSOR_KERN_DMUL2_I_I_I_H

#include "../kern_dmul2.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ c_i = c_i + a_i b_i \f$
    \tparam LA Linear algebra.

    \ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dmul2_i_i_i : public kernel_base<2, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni;
    size_t m_sia, m_sib, m_sic;

public:
    virtual ~kern_dmul2_i_i_i() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<2, 1> &r);

    static kernel_base<2, 1> *match(const kern_dmul2<LA> &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_I_I_I_H
