#include <libtensor/btod/scalar_transf_double.h>
#include "../so_symmetrize.h"
#include "so_symmetrize_impl.h"

namespace libtensor {

template class so_symmetrize<1, double>;
template class so_symmetrize<2, double>;
template class so_symmetrize<3, double>;
template class so_symmetrize<4, double>;
template class so_symmetrize<5, double>;
template class so_symmetrize<6, double>;

} // namespace libtensor


