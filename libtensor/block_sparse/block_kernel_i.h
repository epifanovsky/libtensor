#ifndef BLOCK_KERNEL_I_H
#define BLOCK_KERNEL_I_H

#include <vector>
#include "../core/sequence.h"

namespace libtensor {

typedef std::vector<size_t> dim_list;

template<size_t M,size_t N,typename T = double>
class block_kernel_i {
public:
    virtual void operator()(const sequence<M, T*>& output_ptrs, 
                            const sequence<N, T*>& input_ptrs,
                            const sequence<M, dim_list>& output_dims,
                            const sequence<N, dim_list>& input_dims) = 0;

    //TODO! May not need this anymore...
    //Used to populate instance variables of objects that contain this
    //Necessary because this is an abstract base class
    virtual block_kernel_i<M,N,T>* clone() const = 0;
};

} // namespace libtensor

#endif /* BLOCK_KERNEL_I_H */
