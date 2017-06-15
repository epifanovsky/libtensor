#include "../kern_mul1.h"

namespace libtensor {


template<typename T>
const char *kern_mul1<T>::k_clazz = "kern_mul1<T>";


template<typename T>
void kern_mul1<T>::run(void*, const loop_registers_x<1, 1, T> &r) {

    r.m_ptrb[0][0] *= r.m_ptra[0][0] * m_d;
}

template<typename T>
kernel_base<linalg, 1, 1, T> *kern_mul1<T>::match(T d, list_t &in,
    list_t &out) {

    kern_mul1<T> zz;
    zz.m_d = d;

    return new kern_mul1<T>(zz);
}

template class kern_mul1<double>;
template class kern_mul1<float>;

} // namespace libtensor
