#include <libtensor/btod/scalar_transf_double.h>
#include "../so_apply_se_perm.h"
#include "so_apply_se_perm_impl.h"

namespace libtensor {

template
class symmetry_operation_impl< so_apply<1, double>, se_perm<1, double> >;
template
class symmetry_operation_impl< so_apply<2, double>, se_perm<2, double> >;
template
class symmetry_operation_impl< so_apply<3, double>, se_perm<3, double> >;
template
class symmetry_operation_impl< so_apply<4, double>, se_perm<4, double> >;
template
class symmetry_operation_impl< so_apply<5, double>, se_perm<5, double> >;
template
class symmetry_operation_impl< so_apply<6, double>, se_perm<6, double> >;

} // namespace libtensor


