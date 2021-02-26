#include "../kern_divadd1.h"

namespace libtensor {

template<typename T>
const char *kern_divadd1<T>::k_clazz = "kern_divadd1";


template<typename T>
void kern_divadd1<T>::run(void*, const loop_registers_x<1, 1, T> &r) {

    r.m_ptrb[0][0] = r.m_ptrb[0][0] + (r.m_ptrb[0][0] * m_d) / r.m_ptra[0][0];
}


template<typename T>
kernel_base<linalg, 1, 1, T> *kern_divadd1<T>::match(T d, list_t &in,
    list_t &out) {

    kern_divadd1<T> zz;
    zz.m_d = d;

    return new kern_divadd1(zz);
}

template class kern_divadd1<double>;
template class kern_divadd1<float>;

} // namespace libtensor
