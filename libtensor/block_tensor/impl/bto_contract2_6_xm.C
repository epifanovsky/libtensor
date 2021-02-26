#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_set.h>
#include "bto_contract2_impl.h"
#include "bto_contract2_xm_impl.h"

namespace libtensor {


template class gen_bto_contract2< 0, 6, 1, bto_traits<double>,
    bto_contract2_xm<0, 6, 1, double> >;
template class gen_bto_contract2< 1, 5, 0, bto_traits<double>,
    bto_contract2_xm<1, 5, 0, double> >;
template class gen_bto_contract2< 1, 5, 1, bto_traits<double>,
    bto_contract2_xm<1, 5, 1, double> >;
template class gen_bto_contract2< 1, 5, 2, bto_traits<double>,
    bto_contract2_xm<1, 5, 2, double> >;
template class gen_bto_contract2< 1, 5, 3, bto_traits<double>,
    bto_contract2_xm<1, 5, 3, double> >;
template class gen_bto_contract2< 2, 4, 0, bto_traits<double>,
    bto_contract2_xm<2, 4, 0, double> >;
template class gen_bto_contract2< 2, 4, 1, bto_traits<double>,
    bto_contract2_xm<2, 4, 1, double> >;
template class gen_bto_contract2< 2, 4, 2, bto_traits<double>,
    bto_contract2_xm<2, 4, 2, double> >;
template class gen_bto_contract2< 2, 4, 3, bto_traits<double>,
    bto_contract2_xm<2, 4, 3, double> >;
template class gen_bto_contract2< 2, 4, 4, bto_traits<double>,
    bto_contract2_xm<2, 4, 4, double> >;
template class gen_bto_contract2< 3, 3, 0, bto_traits<double>,
    bto_contract2_xm<3, 3, 0, double> >;
template class gen_bto_contract2< 3, 3, 1, bto_traits<double>,
    bto_contract2_xm<3, 3, 1, double> >;
template class gen_bto_contract2< 3, 3, 2, bto_traits<double>,
    bto_contract2_xm<3, 3, 2, double> >;
template class gen_bto_contract2< 3, 3, 3, bto_traits<double>,
    bto_contract2_xm<3, 3, 3, double> >;
template class gen_bto_contract2< 3, 3, 4, bto_traits<double>,
    bto_contract2_xm<3, 3, 4, double> >;
template class gen_bto_contract2< 3, 3, 5, bto_traits<double>,
    bto_contract2_xm<3, 3, 5, double> >;
template class gen_bto_contract2< 4, 2, 0, bto_traits<double>,
    bto_contract2_xm<4, 2, 0, double> >;
template class gen_bto_contract2< 4, 2, 1, bto_traits<double>,
    bto_contract2_xm<4, 2, 1, double> >;
template class gen_bto_contract2< 4, 2, 2, bto_traits<double>,
    bto_contract2_xm<4, 2, 2, double> >;
template class gen_bto_contract2< 4, 2, 3, bto_traits<double>,
    bto_contract2_xm<4, 2, 3, double> >;
template class gen_bto_contract2< 4, 2, 4, bto_traits<double>,
    bto_contract2_xm<4, 2, 4, double> >;
template class gen_bto_contract2< 5, 1, 0, bto_traits<double>,
    bto_contract2_xm<5, 1, 0, double> >;
template class gen_bto_contract2< 5, 1, 1, bto_traits<double>,
    bto_contract2_xm<5, 1, 1, double> >;
template class gen_bto_contract2< 5, 1, 2, bto_traits<double>,
    bto_contract2_xm<5, 1, 2, double> >;
template class gen_bto_contract2< 5, 1, 3, bto_traits<double>,
    bto_contract2_xm<5, 1, 3, double> >;
template class gen_bto_contract2< 6, 0, 1, bto_traits<double>,
    bto_contract2_xm<6, 0, 1, double> >;


template class gen_bto_contract2< 0, 6, 1, bto_traits<float>,
    bto_contract2_xm<0, 6, 1, float> >;
template class gen_bto_contract2< 1, 5, 0, bto_traits<float>,
    bto_contract2_xm<1, 5, 0, float> >;
template class gen_bto_contract2< 1, 5, 1, bto_traits<float>,
    bto_contract2_xm<1, 5, 1, float> >;
template class gen_bto_contract2< 1, 5, 2, bto_traits<float>,
    bto_contract2_xm<1, 5, 2, float> >;
template class gen_bto_contract2< 1, 5, 3, bto_traits<float>,
    bto_contract2_xm<1, 5, 3, float> >;
template class gen_bto_contract2< 2, 4, 0, bto_traits<float>,
    bto_contract2_xm<2, 4, 0, float> >;
template class gen_bto_contract2< 2, 4, 1, bto_traits<float>,
    bto_contract2_xm<2, 4, 1, float> >;
template class gen_bto_contract2< 2, 4, 2, bto_traits<float>,
    bto_contract2_xm<2, 4, 2, float> >;
template class gen_bto_contract2< 2, 4, 3, bto_traits<float>,
    bto_contract2_xm<2, 4, 3, float> >;
template class gen_bto_contract2< 2, 4, 4, bto_traits<float>,
    bto_contract2_xm<2, 4, 4, float> >;
template class gen_bto_contract2< 3, 3, 0, bto_traits<float>,
    bto_contract2_xm<3, 3, 0, float> >;
template class gen_bto_contract2< 3, 3, 1, bto_traits<float>,
    bto_contract2_xm<3, 3, 1, float> >;
template class gen_bto_contract2< 3, 3, 2, bto_traits<float>,
    bto_contract2_xm<3, 3, 2, float> >;
template class gen_bto_contract2< 3, 3, 3, bto_traits<float>,
    bto_contract2_xm<3, 3, 3, float> >;
template class gen_bto_contract2< 3, 3, 4, bto_traits<float>,
    bto_contract2_xm<3, 3, 4, float> >;
template class gen_bto_contract2< 3, 3, 5, bto_traits<float>,
    bto_contract2_xm<3, 3, 5, float> >;
template class gen_bto_contract2< 4, 2, 0, bto_traits<float>,
    bto_contract2_xm<4, 2, 0, float> >;
template class gen_bto_contract2< 4, 2, 1, bto_traits<float>,
    bto_contract2_xm<4, 2, 1, float> >;
template class gen_bto_contract2< 4, 2, 2, bto_traits<float>,
    bto_contract2_xm<4, 2, 2, float> >;
template class gen_bto_contract2< 4, 2, 3, bto_traits<float>,
    bto_contract2_xm<4, 2, 3, float> >;
template class gen_bto_contract2< 4, 2, 4, bto_traits<float>,
    bto_contract2_xm<4, 2, 4, float> >;
template class gen_bto_contract2< 5, 1, 0, bto_traits<float>,
    bto_contract2_xm<5, 1, 0, float> >;
template class gen_bto_contract2< 5, 1, 1, bto_traits<float>,
    bto_contract2_xm<5, 1, 1, float> >;
template class gen_bto_contract2< 5, 1, 2, bto_traits<float>,
    bto_contract2_xm<5, 1, 2, float> >;
template class gen_bto_contract2< 5, 1, 3, bto_traits<float>,
    bto_contract2_xm<5, 1, 3, float> >;
template class gen_bto_contract2< 6, 0, 1, bto_traits<float>,
    bto_contract2_xm<6, 0, 1, float> >;

template class bto_contract2_xm<0, 6, 1, double>;
template class bto_contract2_xm<0, 6, 2, double>;
template class bto_contract2_xm<1, 5, 0, double>;
template class bto_contract2_xm<1, 5, 1, double>;
template class bto_contract2_xm<1, 5, 2, double>;
template class bto_contract2_xm<1, 5, 3, double>;
template class bto_contract2_xm<2, 4, 0, double>;
template class bto_contract2_xm<2, 4, 1, double>;
template class bto_contract2_xm<2, 4, 2, double>;
template class bto_contract2_xm<2, 4, 3, double>;
template class bto_contract2_xm<2, 4, 4, double>;
template class bto_contract2_xm<3, 3, 0, double>;
template class bto_contract2_xm<3, 3, 1, double>;
template class bto_contract2_xm<3, 3, 2, double>;
template class bto_contract2_xm<3, 3, 3, double>;
template class bto_contract2_xm<3, 3, 4, double>;
template class bto_contract2_xm<3, 3, 5, double>;
template class bto_contract2_xm<4, 2, 0, double>;
template class bto_contract2_xm<4, 2, 1, double>;
template class bto_contract2_xm<4, 2, 2, double>;
template class bto_contract2_xm<4, 2, 3, double>;
template class bto_contract2_xm<4, 2, 4, double>;
template class bto_contract2_xm<5, 1, 0, double>;
template class bto_contract2_xm<5, 1, 1, double>;
template class bto_contract2_xm<5, 1, 2, double>;
template class bto_contract2_xm<5, 1, 3, double>;
template class bto_contract2_xm<6, 0, 1, double>;
template class bto_contract2_xm<6, 0, 2, double>;

template class bto_contract2_xm<0, 6, 1, float>;
template class bto_contract2_xm<0, 6, 2, float>;
template class bto_contract2_xm<1, 5, 0, float>;
template class bto_contract2_xm<1, 5, 1, float>;
template class bto_contract2_xm<1, 5, 2, float>;
template class bto_contract2_xm<1, 5, 3, float>;
template class bto_contract2_xm<2, 4, 0, float>;
template class bto_contract2_xm<2, 4, 1, float>;
template class bto_contract2_xm<2, 4, 2, float>;
template class bto_contract2_xm<2, 4, 3, float>;
template class bto_contract2_xm<2, 4, 4, float>;
template class bto_contract2_xm<3, 3, 0, float>;
template class bto_contract2_xm<3, 3, 1, float>;
template class bto_contract2_xm<3, 3, 2, float>;
template class bto_contract2_xm<3, 3, 3, float>;
template class bto_contract2_xm<3, 3, 4, float>;
template class bto_contract2_xm<3, 3, 5, float>;
template class bto_contract2_xm<4, 2, 0, float>;
template class bto_contract2_xm<4, 2, 1, float>;
template class bto_contract2_xm<4, 2, 2, float>;
template class bto_contract2_xm<4, 2, 3, float>;
template class bto_contract2_xm<4, 2, 4, float>;
template class bto_contract2_xm<5, 1, 0, float>;
template class bto_contract2_xm<5, 1, 1, float>;
template class bto_contract2_xm<5, 1, 2, float>;
template class bto_contract2_xm<5, 1, 3, float>;
template class bto_contract2_xm<6, 0, 1, float>;
template class bto_contract2_xm<6, 0, 2, float>;

} // namespace libtensor
