#include <libtensor/core/scalar_transf_double.h>
#include "../so_permute_se_part.h"
#include "so_permute_se_part_impl.h"

namespace libtensor {


template class symmetry_operation_impl< so_permute<1, double>,
    se_part<1, double> >;
template class symmetry_operation_impl< so_permute<2, double>,
    se_part<2, double> >;
template class symmetry_operation_impl< so_permute<3, double>,
    se_part<3, double> >;
template class symmetry_operation_impl< so_permute<4, double>,
    se_part<4, double> >;
template class symmetry_operation_impl< so_permute<5, double>,
    se_part<5, double> >;
template class symmetry_operation_impl< so_permute<6, double>,
    se_part<6, double> >;
template class symmetry_operation_impl< so_permute<7, double>,
    se_part<7, double> >;
template class symmetry_operation_impl< so_permute<8, double>,
    se_part<8, double> >;


} // namespace libtensor


