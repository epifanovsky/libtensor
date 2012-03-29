#include <libtensor/btod/scalar_transf_double.h>
#include "../so_apply_se_label.h"
#include "so_apply_se_label_impl.h"

namespace libtensor {

template
class symmetry_operation_impl< so_apply<1, double>, se_label<1, double> >;
template
class symmetry_operation_impl< so_apply<2, double>, se_label<2, double> >;
template
class symmetry_operation_impl< so_apply<3, double>, se_label<3, double> >;
template
class symmetry_operation_impl< so_apply<4, double>, se_label<4, double> >;
template
class symmetry_operation_impl< so_apply<5, double>, se_label<5, double> >;
template
class symmetry_operation_impl< so_apply<6, double>, se_label<6, double> >;

} // namespace libtensor


