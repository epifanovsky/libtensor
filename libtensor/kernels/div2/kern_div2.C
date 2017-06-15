#include "../kern_div2.h"

namespace libtensor {


template<typename T>
const char *kern_div2<T>::k_clazz = "kern_div2";


template<typename T>
void kern_div2<T>::run(void*, const loop_registers_x<2, 1, T> &r) {

    r.m_ptrb[0][0] += m_d * r.m_ptra[0][0] / r.m_ptra[1][0];

}


template<typename T>
kernel_base<linalg, 2, 1, T> *kern_div2<T>::match(T d, list_t &in,
    list_t &out) {

    kern_div2<T> zz;
    zz.m_d = d;

    return new kern_div2(zz);
}

template class kern_div2<double>;
template class kern_div2<float>;

} // namespace libtensor
