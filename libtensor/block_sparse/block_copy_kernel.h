#ifndef BLOCK_COPY_KERNEL_H
#define BLOCK_COPY_KERNEL_H

#include <stdlib.h>
#include "block_kernel_i.h"

namespace libtensor {

template<typename T>
class block_copy_kernel : public block_kernel_i<1,1,T> {
public: 
    static const char *k_clazz; //!< Class name
public:
    void operator()(const sequence<1, T*>& output_ptrs, 
                    const sequence<1, const T*>& input_ptrs,
                    const sequence<1, dim_list>& output_dims,
                    const sequence<1, dim_list >& input_dims);

    void operator()(const sequence<1, T*>& output_ptrs, 
                    const sequence<1, const T*>& input_ptrs,
                    const sequence<1, dim_list>& output_dims,
                    const sequence<1, dim_list>& input_dims,
                    const sequence<1, dim_list>& output_inds,
                    const sequence<1, dim_list>& input_inds) { (*this)(output_ptrs,input_ptrs,output_dims,input_dims); }

    //Default constructor
    block_copy_kernel() { };
    //Copy constructor
    block_copy_kernel(const block_copy_kernel<T>& rhs) { };
};

template<typename T>
const char *block_copy_kernel<T>::k_clazz = "block_copy_kernel<T>";

template<typename T>
void block_copy_kernel<T>::operator()(const sequence<1, T*>& output_ptrs, 
                                      const sequence<1, const T*>& input_ptrs,
                                      const sequence<1, dim_list>& output_dims,
                                      const sequence<1, dim_list >& input_dims)
{
    const dim_list& ob_dims = output_dims[0];
    const dim_list& ib_dims = input_dims[0];

    if(ob_dims.size() != ib_dims.size())
    {
        throw bad_parameter(g_ns, k_clazz,"operator(...)",
                __FILE__, __LINE__, "output and input blocks do not have the same dimensionality");
    }

    for(int i = 0; i < ob_dims.size(); ++i)
    {
        if(ob_dims[i] != ib_dims[i])
        {
            throw bad_parameter(g_ns, k_clazz, "operator(...)",
                    __FILE__, __LINE__, "output and input dimensions do not match");
        }
    }

    size_t block_size = 1;
    for(int i = 0; i < ob_dims.size(); ++i)
    {
        block_size *= ob_dims[i];
    }

    memcpy(output_ptrs[0],input_ptrs[0],block_size*sizeof(T));
}

} // namespace libtensor

#endif /* BLOCK_COPY_KERNEL_H */
