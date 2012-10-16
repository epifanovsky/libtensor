#ifndef LIBTENSOR_KERN_APPLY_H
#define LIBTENSOR_KERN_APPLY_H

#include <libtensor/linalg/linalg.h>
#include "kernel_base.h"

namespace libtensor {


template<typename Functor>
class kern_apply : public kernel_base<linalg, 1, 1> {
public:
    static const char *k_clazz; //!< Kernel name

private:
    Functor *m_fn; //!< Functor
    double m_c1, m_c2;

public:
    virtual ~kern_apply() { }

    virtual const char *get_name() const {
        return k_clazz;
    }

    virtual void run(void *, const loop_registers<1, 1> &r);

    static kernel_base<linalg, 1, 1> *match(Functor &fn,
            double c1, double c2, list_t &in, list_t &out);
};


template<typename Functor>
const char *kern_apply<Functor>::k_clazz = "kern_apply";


template<typename Functor>
void kern_apply<Functor>::run(void *, const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] = m_c2 * (*m_fn)(m_c1 * r.m_ptra[0][0]);
}


template<typename Functor>
kernel_base<linalg, 1, 1> *kern_apply<Functor>::match(Functor &fn,
        double c1, double c2, list_t &in, list_t &out) {

    kernel_base<linalg, 1, 1> *kern = 0;

    kern_apply zz;
    zz.m_fn = &fn;
    zz.m_c1 = c1;
    zz.m_c2 = c2;

    return new kern_apply(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_APPLY_H
