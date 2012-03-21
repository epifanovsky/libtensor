#include "../so_symmetrize3_se_part.h"
#include "so_symmetrize3_se_part_impl.h"

namespace libtensor {

template class symmetry_operation_impl< so_symmetrize3<1, double>,
    se_part<1, double> >;
template class symmetry_operation_impl< so_symmetrize3<2, double>,
    se_part<2, double> >;
template class symmetry_operation_impl< so_symmetrize3<3, double>,
    se_part<3, double> >;
template class symmetry_operation_impl< so_symmetrize3<4, double>,
    se_part<4, double> >;
template class symmetry_operation_impl< so_symmetrize3<5, double>,
    se_part<5, double> >;
template class symmetry_operation_impl< so_symmetrize3<6, double>,
    se_part<6, double> >;

} // namespace libtensor


