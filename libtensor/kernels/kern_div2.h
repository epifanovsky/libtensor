#ifndef LIBTENSOR_KERN_DIV2_H
#define LIBTENSOR_KERN_DIV2_H

#include <libtensor/linalg/linalg.h>
#include "loop_list_node.h"
#include "kernel_base.h"

namespace libtensor {


/** \brief Generic division kernel (T)

    This kernel divides two multidimensional arrays with optional scaling:
    \f[
        c = c + d \frac{a}{b}
    \f]
    a, b, c are arrays, d is a scaling factors.

    \ingroup libtensor_kernels
 **/
template<typename T>
class kern_div2 : public kernel_base<linalg, 2, 1, T> {
public:
    typedef std::list< loop_list_node<2, 1> > list_t;
    static const char *k_clazz; //!< Kernel name

private:
    T m_d;

public:
    virtual ~kern_div2() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(void*, const loop_registers_x<2, 1, T> &r);

    static kernel_base<linalg, 2, 1, T> *match(T d, list_t &in, list_t &out);

};

using kern_ddiv2 = kern_div2<double>;

} // namespace libtensor

#endif // LIBTENSOR_KERN_DIV2_H
