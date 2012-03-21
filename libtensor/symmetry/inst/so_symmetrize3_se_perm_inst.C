#include "../so_symmetrize3_se_perm.h"
#include "so_symmetrize3_se_perm_impl.h"

namespace libtensor {

template class symmetry_operation_impl< so_symmetrize3<1, double>,
    se_perm<1, double> >;
template class symmetry_operation_impl< so_symmetrize3<2, double>,
    se_perm<2, double> >;
template class symmetry_operation_impl< so_symmetrize3<3, double>,
    se_perm<3, double> >;
template class symmetry_operation_impl< so_symmetrize3<4, double>,
    se_perm<4, double> >;
template class symmetry_operation_impl< so_symmetrize3<5, double>,
    se_perm<5, double> >;
template class symmetry_operation_impl< so_symmetrize3<6, double>,
    se_perm<6, double> >;

} // namespace libtensor


