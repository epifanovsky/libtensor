#include "to_dirsum_dims_impl.h"

namespace libtensor {


template class to_dirsum_dims<1, 1>;

template class to_dirsum_dims<1, 2>;
template class to_dirsum_dims<2, 1>;

template class to_dirsum_dims<1, 3>;
template class to_dirsum_dims<2, 2>;
template class to_dirsum_dims<3, 1>;

template class to_dirsum_dims<1, 4>;
template class to_dirsum_dims<2, 3>;
template class to_dirsum_dims<3, 2>;
template class to_dirsum_dims<4, 1>;

template class to_dirsum_dims<1, 5>;
template class to_dirsum_dims<2, 4>;
template class to_dirsum_dims<3, 3>;
template class to_dirsum_dims<4, 2>;
template class to_dirsum_dims<5, 1>;

template class to_dirsum_dims<1, 6>;
template class to_dirsum_dims<2, 5>;
template class to_dirsum_dims<3, 4>;
template class to_dirsum_dims<4, 3>;
template class to_dirsum_dims<5, 2>;
template class to_dirsum_dims<6, 1>;

template class to_dirsum_dims<1, 7>;
template class to_dirsum_dims<2, 6>;
template class to_dirsum_dims<3, 5>;
template class to_dirsum_dims<4, 4>;
template class to_dirsum_dims<5, 3>;
template class to_dirsum_dims<6, 2>;
template class to_dirsum_dims<7, 1>;


} // namespace libtensor
