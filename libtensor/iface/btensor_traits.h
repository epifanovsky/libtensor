#ifndef LIBTENSOR_BTENSOR_TRAITS_H
#define LIBTENSOR_BTENSOR_TRAITS_H

#include "../core/allocator.h"

namespace libtensor {


template<typename T>
struct btensor_traits {
    typedef T element_t;
    typedef allocator<T> allocator_t;
};


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_TRAITS_H
