#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_set.h>
#include "btod_contract2_impl.h"

namespace libtensor {


template<>
struct btod_contract2_clazz<0, 2, 1> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<0, 2, 2> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<0, 2, 3> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<0, 2, 4> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 1, 0> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 1, 1> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 1, 2> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 1, 3> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 1, 4> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<1, 1, 5> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<2, 0, 1> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<2, 0, 2> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<2, 0, 3> {
    static const char *k_clazz;
};
template<>
struct btod_contract2_clazz<2, 0, 4> {
    static const char *k_clazz;
};


const char *btod_contract2_clazz<0, 2, 1>::k_clazz = "btod_contract2<0, 2, 1>";
const char *btod_contract2_clazz<0, 2, 2>::k_clazz = "btod_contract2<0, 2, 2>";
const char *btod_contract2_clazz<0, 2, 3>::k_clazz = "btod_contract2<0, 2, 3>";
const char *btod_contract2_clazz<0, 2, 4>::k_clazz = "btod_contract2<0, 2, 4>";
const char *btod_contract2_clazz<1, 1, 0>::k_clazz = "btod_contract2<1, 1, 0>";
const char *btod_contract2_clazz<1, 1, 1>::k_clazz = "btod_contract2<1, 1, 1>";
const char *btod_contract2_clazz<1, 1, 2>::k_clazz = "btod_contract2<1, 1, 2>";
const char *btod_contract2_clazz<1, 1, 3>::k_clazz = "btod_contract2<1, 1, 3>";
const char *btod_contract2_clazz<1, 1, 4>::k_clazz = "btod_contract2<1, 1, 4>";
const char *btod_contract2_clazz<1, 1, 5>::k_clazz = "btod_contract2<1, 1, 5>";
const char *btod_contract2_clazz<2, 0, 1>::k_clazz = "btod_contract2<2, 0, 1>";
const char *btod_contract2_clazz<2, 0, 2>::k_clazz = "btod_contract2<2, 0, 2>";
const char *btod_contract2_clazz<2, 0, 3>::k_clazz = "btod_contract2<2, 0, 3>";
const char *btod_contract2_clazz<2, 0, 4>::k_clazz = "btod_contract2<2, 0, 4>";


template class gen_bto_contract2< 0, 2, 1, btod_traits,
    btod_contract2<0, 2, 1> >;
template class gen_bto_contract2< 0, 2, 2, btod_traits,
    btod_contract2<0, 2, 2> >;
template class gen_bto_contract2< 0, 2, 3, btod_traits,
    btod_contract2<0, 2, 3> >;
template class gen_bto_contract2< 0, 2, 4, btod_traits,
    btod_contract2<0, 2, 4> >;
template class gen_bto_contract2< 1, 1, 0, btod_traits,
    btod_contract2<1, 1, 0> >;
template class gen_bto_contract2< 1, 1, 1, btod_traits,
    btod_contract2<1, 1, 1> >;
template class gen_bto_contract2< 1, 1, 2, btod_traits,
    btod_contract2<1, 1, 2> >;
template class gen_bto_contract2< 1, 1, 3, btod_traits,
    btod_contract2<1, 1, 3> >;
template class gen_bto_contract2< 1, 1, 4, btod_traits,
    btod_contract2<1, 1, 4> >;
template class gen_bto_contract2< 1, 1, 5, btod_traits,
    btod_contract2<1, 1, 5> >;
template class gen_bto_contract2< 2, 0, 1, btod_traits,
    btod_contract2<2, 0, 1> >;
template class gen_bto_contract2< 2, 0, 2, btod_traits,
    btod_contract2<2, 0, 2> >;
template class gen_bto_contract2< 2, 0, 3, btod_traits,
    btod_contract2<2, 0, 3> >;
template class gen_bto_contract2< 2, 0, 4, btod_traits,
    btod_contract2<2, 0, 4> >;


template class btod_contract2<0, 2, 1>;
template class btod_contract2<0, 2, 2>;
template class btod_contract2<0, 2, 3>;
template class btod_contract2<0, 2, 4>;
template class btod_contract2<1, 1, 0>;
template class btod_contract2<1, 1, 1>;
template class btod_contract2<1, 1, 2>;
template class btod_contract2<1, 1, 3>;
template class btod_contract2<1, 1, 4>;
template class btod_contract2<1, 1, 5>;
template class btod_contract2<2, 0, 1>;
template class btod_contract2<2, 0, 2>;
template class btod_contract2<2, 0, 3>;
template class btod_contract2<2, 0, 4>;


} // namespace libtensor
