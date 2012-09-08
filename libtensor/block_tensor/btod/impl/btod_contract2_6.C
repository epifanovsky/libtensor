#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_set.h>
#include "btod_contract2_impl.h"

namespace libtensor {


template<>
struct btod_contract2_clazz<1, 5, 0> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 5, 1> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<2, 4, 0> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<2, 4, 1> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<2, 4, 2> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<3, 3, 0> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<3, 3, 1> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<3, 3, 2> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<3, 3, 3> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<4, 2, 0> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<4, 2, 1> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<4, 2, 2> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<5, 1, 0> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<5, 1, 1> {
    static const char *k_clazz;
};


const char *btod_contract2_clazz<1, 5, 0>::k_clazz = "btod_contract2<1, 5, 0>";
const char *btod_contract2_clazz<1, 5, 1>::k_clazz = "btod_contract2<1, 5, 1>";
const char *btod_contract2_clazz<2, 4, 0>::k_clazz = "btod_contract2<2, 4, 0>";
const char *btod_contract2_clazz<2, 4, 1>::k_clazz = "btod_contract2<2, 4, 1>";
const char *btod_contract2_clazz<2, 4, 2>::k_clazz = "btod_contract2<2, 4, 2>";
const char *btod_contract2_clazz<3, 3, 0>::k_clazz = "btod_contract2<3, 3, 0>";
const char *btod_contract2_clazz<3, 3, 1>::k_clazz = "btod_contract2<3, 3, 1>";
const char *btod_contract2_clazz<3, 3, 2>::k_clazz = "btod_contract2<3, 3, 2>";
const char *btod_contract2_clazz<3, 3, 3>::k_clazz = "btod_contract2<3, 3, 3>";
const char *btod_contract2_clazz<4, 2, 0>::k_clazz = "btod_contract2<4, 2, 0>";
const char *btod_contract2_clazz<4, 2, 1>::k_clazz = "btod_contract2<4, 2, 1>";
const char *btod_contract2_clazz<4, 2, 2>::k_clazz = "btod_contract2<4, 2, 2>";
const char *btod_contract2_clazz<5, 1, 0>::k_clazz = "btod_contract2<5, 1, 0>";
const char *btod_contract2_clazz<5, 1, 1>::k_clazz = "btod_contract2<5, 1, 1>";


template class btod_contract2<1, 5, 0>;
template class btod_contract2<1, 5, 1>;
template class btod_contract2<2, 4, 0>;
template class btod_contract2<2, 4, 1>;
template class btod_contract2<2, 4, 2>;
template class btod_contract2<3, 3, 0>;
template class btod_contract2<3, 3, 1>;
template class btod_contract2<3, 3, 2>;
template class btod_contract2<3, 3, 3>;
template class btod_contract2<4, 2, 0>;
template class btod_contract2<4, 2, 1>;
template class btod_contract2<4, 2, 2>;
template class btod_contract2<5, 1, 0>;
template class btod_contract2<5, 1, 1>;


} // namespace libtensor
