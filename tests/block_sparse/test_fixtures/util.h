#ifndef UTIL_H
#define UTIL_H

#include <libtensor/core/sequence.h>

namespace libtensor {

template<size_t N>
std::vector< sequence<N,size_t> > get_sig_blocks(const size_t arr[][N],size_t n_entries)
{
    std::vector< sequence<N,size_t> > sig_blocks(n_entries);
    for(size_t i = 0; i < n_entries; ++i)
        for(size_t j = 0; j < N; ++j) sig_blocks[i][j] = arr[i][j];
    return sig_blocks;
}

} // namespace libtensor
#endif /* UTIL_H */
