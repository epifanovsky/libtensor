#ifndef LIBTENSOR_KERN_APPLY_H
#define LIBTENSOR_KERN_APPLY_H

#include <libtensor/linalg/linalg.h>
#include "kernel_base.h"

namespace libtensor {


/** \brief Generic function application kernel (T)
    \tparam LA Linear algebra.

    This kernel applies a function to a multidimensional arrays with optional
    scaling:
    \f[
        b = c_2 f(c_1 a)
    \f]
    a, b are arrays, c_1, c_2 are scaling factors.

    \ingroup libtensor_kernels
 **/
template<typename Functor, typename T>
class kern_apply : public kernel_base<linalg, 1, 1, T> {
public:
    static const char *k_clazz; //!< Kernel name
    typedef std::list< loop_list_node<1, 1> > list_t;

private:
    Functor *m_fn; //!< Functor
    T m_c1, m_c2;

public:
    virtual ~kern_apply() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(void *, const loop_registers_x<1, 1, T> &r);

    static kernel_base<linalg, 1, 1, T> *match(Functor &fn,
            T c1, T c2, list_t &in, list_t &out);
};


template<typename Functor, typename T>
const char *kern_apply<Functor, T>::k_clazz = "kern_apply";


template<typename Functor, typename T>
void kern_apply<Functor, T>::run(void *, const loop_registers_x<1, 1, T> &r) {

    r.m_ptrb[0][0] = m_c2 * (*m_fn)(m_c1 * r.m_ptra[0][0]);
}


template<typename Functor, typename T>
kernel_base<linalg, 1, 1, T> *kern_apply<Functor, T>::match(Functor &fn,
        T c1, T c2, list_t &in, list_t &out) {

    kern_apply<Functor, T> zz;
    zz.m_fn = &fn;
    zz.m_c1 = c1;
    zz.m_c2 = c2;

    return new kern_apply(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_APPLY_H
