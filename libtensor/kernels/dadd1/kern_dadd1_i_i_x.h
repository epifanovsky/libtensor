#ifndef LIBTENSOR_KERN_DADD1_I_I_X_H
#define LIBTENSOR_KERN_DADD1_I_I_X_H

#include "../kern_dadd1.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ b_i = b_i + a_i d \f$

     \ingroup libtensor_kernels
 **/
class kern_dadd1_i_i_x : public kernel_base<1, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    double m_d;
    size_t m_ni;
    size_t m_sia, m_sib;

public:
    virtual ~kern_dadd1_i_i_x() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(const loop_registers<1, 1> &r);

    static kernel_base<1, 1> *match(const kern_dadd1 &z,
        list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DADD1_I_I_X_H
