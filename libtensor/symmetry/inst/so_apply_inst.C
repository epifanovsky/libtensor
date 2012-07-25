#include <libtensor/core/scalar_transf_double.h>
#include "../so_apply.h"
#include "so_apply_impl.h"

namespace libtensor {

template class so_apply<1, double>;
template class so_apply<2, double>;
template class so_apply<3, double>;
template class so_apply<4, double>;
template class so_apply<5, double>;
template class so_apply<6, double>;

} // namespace libtensor


