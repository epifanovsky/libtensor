#include <libtensor/core/scalar_transf_double.h>
#include "../so_permute.h"
#include "so_permute_impl.h"

namespace libtensor {


template class so_permute<1, double>;
template class so_permute<2, double>;
template class so_permute<3, double>;
template class so_permute<4, double>;
template class so_permute<5, double>;
template class so_permute<6, double>;
template class so_permute<7, double>;
template class so_permute<8, double>;

template class so_permute<1, float>;
template class so_permute<2, float>;
template class so_permute<3, float>;
template class so_permute<4, float>;
template class so_permute<5, float>;
template class so_permute<6, float>;
template class so_permute<7, float>;
template class so_permute<8, float>;

} // namespace libtensor


