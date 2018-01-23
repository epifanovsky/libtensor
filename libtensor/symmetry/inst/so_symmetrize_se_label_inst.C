#include <libtensor/core/scalar_transf_double.h>
#include "../so_symmetrize_se_label.h"
#include "so_symmetrize_se_label_impl.h"

namespace libtensor {


template class symmetry_operation_impl< so_symmetrize<1, double>,
    se_label<1, double> >;
template class symmetry_operation_impl< so_symmetrize<2, double>,
    se_label<2, double> >;
template class symmetry_operation_impl< so_symmetrize<3, double>,
    se_label<3, double> >;
template class symmetry_operation_impl< so_symmetrize<4, double>,
    se_label<4, double> >;
template class symmetry_operation_impl< so_symmetrize<5, double>,
    se_label<5, double> >;
template class symmetry_operation_impl< so_symmetrize<6, double>,
    se_label<6, double> >;
template class symmetry_operation_impl< so_symmetrize<7, double>,
    se_label<7, double> >;
template class symmetry_operation_impl< so_symmetrize<8, double>,
    se_label<8, double> >;


template class symmetry_operation_impl< so_symmetrize<1, float>,
    se_label<1, float> >;
template class symmetry_operation_impl< so_symmetrize<2, float>,
    se_label<2, float> >;
template class symmetry_operation_impl< so_symmetrize<3, float>,
    se_label<3, float> >;
template class symmetry_operation_impl< so_symmetrize<4, float>,
    se_label<4, float> >;
template class symmetry_operation_impl< so_symmetrize<5, float>,
    se_label<5, float> >;
template class symmetry_operation_impl< so_symmetrize<6, float>,
    se_label<6, float> >;
template class symmetry_operation_impl< so_symmetrize<7, float>,
    se_label<7, float> >;
template class symmetry_operation_impl< so_symmetrize<8, float>,
    se_label<8, float> >;





} // namespace libtensor


