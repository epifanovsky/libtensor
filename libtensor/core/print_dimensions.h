#ifndef LIBTENSOR_PRINT_DIMENSIONS_H
#define LIBTENSOR_PRINT_DIMENSIONS_H

#include <iostream>
#include "dimensions.h"

namespace libtensor {


template<size_t N>
std::ostream &operator<<(std::ostream &os, const dimensions<N> &d) {

    os << "[";
    for(size_t j = 0; j < N; j++) os << d[j] << (j + 1 == N ? "" : ", ");
    os << "]";
    return os;
}


} // namespace libtensor

#endif // LIBTENSOR_PRINT_DIMENSIONS_H
