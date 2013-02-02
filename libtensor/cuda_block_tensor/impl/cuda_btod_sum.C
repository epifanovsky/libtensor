#include <libtensor/core/scalar_transf_double.h>
#include "cuda_btod_sum_impl.h"

namespace libtensor {


template class cuda_btod_sum<1>;
template class cuda_btod_sum<2>;
template class cuda_btod_sum<3>;
template class cuda_btod_sum<4>;
template class cuda_btod_sum<5>;
template class cuda_btod_sum<6>;


} // namespace libtensor
