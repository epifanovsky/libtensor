#ifndef LIBTENSOR_MAGIC_DIMENSIONS_H
#define LIBTENSOR_MAGIC_DIMENSIONS_H

#include "impl/libdivide.h"
#include "dimensions.h"

namespace libtensor {


/** \brief Tensor dimensions class with some magic for fast index transformation

    Index transformations from the regular to the absolute form can become
    a bottleneck in calculations with small blocks. The hotspot is integer
    divisions. This class uses libdivide to alleviate the problem.

    Magic dimensions are created based on regular dimensions with one parameter:
     - incs = true: use tensor modes' linear increments (can be used to convert
       absolute indexes to multi-indexes via abs_index<N>::get_index).
     - incs = false: use tensor dimensions (can be used to find to which block
       an index belongs).

    \sa abs_index, dimensions, index

    \ingroup libtensor_core
 **/
template<size_t N>
class magic_dimensions {
private:
    dimensions<N> m_dims;
    sequence<N, libdivide::divider<uint64_t> > m_magic;

public:
    magic_dimensions(const dimensions<N> &dims, bool incs);

    const dimensions<N> &get_dims() const {
        return m_dims;
    }

    const libdivide::divider<uint64_t> &operator[](size_t i) const;

};


template<>
class magic_dimensions<0> {
public:
    magic_dimensions(const dimensions<0> &dims) {

    }

    const dimensions<0> &get_dims() const {

    }

    const libdivide::divider<uint64_t> &operator[](size_t i) const {
    }

};


} // namespace libtensor

#endif // LIBTENSOR_MAGIC_DIMENSIONS_H
