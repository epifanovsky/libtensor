#include <libtensor/gen_block_tensor/impl/gen_bto_unfold_symmetry_impl.h>
#include "../bto_traits.h"

namespace libtensor {


template class gen_bto_unfold_symmetry< 1, bto_traits<double> >;
template class gen_bto_unfold_symmetry< 2, bto_traits<double> >;
template class gen_bto_unfold_symmetry< 3, bto_traits<double> >;
template class gen_bto_unfold_symmetry< 4, bto_traits<double> >;
template class gen_bto_unfold_symmetry< 5, bto_traits<double> >;
template class gen_bto_unfold_symmetry< 6, bto_traits<double> >;
template class gen_bto_unfold_symmetry< 7, bto_traits<double> >;
template class gen_bto_unfold_symmetry< 8, bto_traits<double> >;

template class gen_bto_unfold_symmetry< 1, bto_traits<float> >;
template class gen_bto_unfold_symmetry< 2, bto_traits<float> >;
template class gen_bto_unfold_symmetry< 3, bto_traits<float> >;
template class gen_bto_unfold_symmetry< 4, bto_traits<float> >;
template class gen_bto_unfold_symmetry< 5, bto_traits<float> >;
template class gen_bto_unfold_symmetry< 6, bto_traits<float> >;
template class gen_bto_unfold_symmetry< 7, bto_traits<float> >;
template class gen_bto_unfold_symmetry< 8, bto_traits<float> >;

} // namespace libtensor
