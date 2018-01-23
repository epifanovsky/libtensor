#include <libtensor/core/scalar_transf_double.h>
#include "../so_symmetrize_se_part.h"
#include "so_symmetrize_se_part_impl.h"

namespace libtensor {


template class symmetry_operation_impl< so_symmetrize<1, double>,
    se_part<1, double> >;
template class symmetry_operation_impl< so_symmetrize<2, double>,
    se_part<2, double> >;
template class symmetry_operation_impl< so_symmetrize<3, double>,
    se_part<3, double> >;
template class symmetry_operation_impl< so_symmetrize<4, double>,
    se_part<4, double> >;
template class symmetry_operation_impl< so_symmetrize<5, double>,
    se_part<5, double> >;
template class symmetry_operation_impl< so_symmetrize<6, double>,
    se_part<6, double> >;
template class symmetry_operation_impl< so_symmetrize<7, double>,
    se_part<7, double> >;
template class symmetry_operation_impl< so_symmetrize<8, double>,
    se_part<8, double> >;


template class symmetry_operation_impl< so_symmetrize<1, float>,
    se_part<1, float> >;
template class symmetry_operation_impl< so_symmetrize<2, float>,
    se_part<2, float> >;
template class symmetry_operation_impl< so_symmetrize<3, float>,
    se_part<3, float> >;
template class symmetry_operation_impl< so_symmetrize<4, float>,
    se_part<4, float> >;
template class symmetry_operation_impl< so_symmetrize<5, float>,
    se_part<5, float> >;
template class symmetry_operation_impl< so_symmetrize<6, float>,
    se_part<6, float> >;
template class symmetry_operation_impl< so_symmetrize<7, float>,
    se_part<7, float> >;
template class symmetry_operation_impl< so_symmetrize<8, float>,
    se_part<8, float> >;





} // namespace libtensor


