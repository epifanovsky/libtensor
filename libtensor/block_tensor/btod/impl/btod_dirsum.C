#include <libtensor/core/scalar_transf_double.h>
#include "btod_dirsum_impl.h"

namespace libtensor {


template<>
struct btod_dirsum_clazz<0, 1> {
    static const char *k_clazz;
};
template<> 
struct btod_dirsum_clazz<0, 2> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<0, 3> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<0, 4> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<0, 5> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<0, 6> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<1, 0> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<1, 1> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<1, 2> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<1, 3> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<1, 4> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<1, 5> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<2, 0> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<2, 1> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<2, 2> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<2, 3> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<2, 4> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<3, 0> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<3, 1> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<3, 2> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<3, 3> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<4, 0> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<4, 1> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<4, 2> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<5, 0> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<5, 1> {
    static const char *k_clazz;
};
template<>
struct btod_dirsum_clazz<6, 0> {
    static const char *k_clazz;
};


const char *btod_dirsum_clazz<0, 1>::k_clazz = "btod_dirsum<0, 1>";
const char *btod_dirsum_clazz<0, 2>::k_clazz = "btod_dirsum<0, 2>";
const char *btod_dirsum_clazz<0, 3>::k_clazz = "btod_dirsum<0, 3>";
const char *btod_dirsum_clazz<0, 4>::k_clazz = "btod_dirsum<0, 4>";
const char *btod_dirsum_clazz<0, 5>::k_clazz = "btod_dirsum<0, 5>";
const char *btod_dirsum_clazz<0, 6>::k_clazz = "btod_dirsum<0, 6>";
const char *btod_dirsum_clazz<1, 0>::k_clazz = "btod_dirsum<1, 0>";
const char *btod_dirsum_clazz<1, 1>::k_clazz = "btod_dirsum<1, 1>";
const char *btod_dirsum_clazz<1, 2>::k_clazz = "btod_dirsum<1, 2>";
const char *btod_dirsum_clazz<1, 3>::k_clazz = "btod_dirsum<1, 3>";
const char *btod_dirsum_clazz<1, 4>::k_clazz = "btod_dirsum<1, 4>";
const char *btod_dirsum_clazz<1, 5>::k_clazz = "btod_dirsum<1, 5>";
const char *btod_dirsum_clazz<2, 0>::k_clazz = "btod_dirsum<2, 0>";
const char *btod_dirsum_clazz<2, 1>::k_clazz = "btod_dirsum<2, 1>";
const char *btod_dirsum_clazz<2, 2>::k_clazz = "btod_dirsum<2, 2>";
const char *btod_dirsum_clazz<2, 3>::k_clazz = "btod_dirsum<2, 3>";
const char *btod_dirsum_clazz<2, 4>::k_clazz = "btod_dirsum<2, 4>";
const char *btod_dirsum_clazz<3, 0>::k_clazz = "btod_dirsum<3, 0>";
const char *btod_dirsum_clazz<3, 1>::k_clazz = "btod_dirsum<3, 1>";
const char *btod_dirsum_clazz<3, 2>::k_clazz = "btod_dirsum<3, 2>";
const char *btod_dirsum_clazz<3, 3>::k_clazz = "btod_dirsum<3, 3>";
const char *btod_dirsum_clazz<4, 0>::k_clazz = "btod_dirsum<4, 0>";
const char *btod_dirsum_clazz<4, 1>::k_clazz = "btod_dirsum<4, 1>";
const char *btod_dirsum_clazz<4, 2>::k_clazz = "btod_dirsum<4, 2>";
const char *btod_dirsum_clazz<5, 0>::k_clazz = "btod_dirsum<5, 0>";
const char *btod_dirsum_clazz<5, 1>::k_clazz = "btod_dirsum<5, 1>";
const char *btod_dirsum_clazz<6, 0>::k_clazz = "btod_dirsum<6, 0>";


template class btod_dirsum<1, 1>;
template class btod_dirsum<1, 2>;
template class btod_dirsum<1, 3>;
template class btod_dirsum<1, 4>;
template class btod_dirsum<1, 5>;
template class btod_dirsum<2, 1>;
template class btod_dirsum<2, 2>;
template class btod_dirsum<2, 3>;
template class btod_dirsum<2, 4>;
template class btod_dirsum<3, 1>;
template class btod_dirsum<3, 2>;
template class btod_dirsum<3, 3>;
template class btod_dirsum<4, 1>;
template class btod_dirsum<4, 2>;
template class btod_dirsum<5, 1>;


} // namespace libtensor
