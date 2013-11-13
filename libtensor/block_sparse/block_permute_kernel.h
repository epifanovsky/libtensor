#ifndef BLOCK_PERMUTE_KERNEL_H
#define BLOCK_PERMUTE_KERNEL_H

#include <map>
#include <numeric>
#include "block_kernel_i.h"
#include "runtime_permutation.h"

namespace libtensor { 

//!!!!! MAPS ARE PASSED IN WITH THE CONVENTION input->output!
template<typename T>
class block_permute_kernel : public block_kernel_i<1,1,T> {
public:
    static const char *k_clazz; //!< Class name
private:
    runtime_permutation m_perm;

    //Recurse internal permutation handler
    void _permute(T* output_ptrs, 
                  const T* input_ptrs,
                  const dim_list& output_dims,
                  const dim_list& input_dims,
                  size_t output_offset = 0,size_t input_offset = 0,size_t level = 0);
public:
    block_permute_kernel(const runtime_permutation& perm) : m_perm(perm) {}

    //Returns a pointer to a copy of this object
    block_kernel_i<1,1,T>* clone() const { return (block_kernel_i<1,1,T>*) new block_permute_kernel(*this); };  

    void operator()(const sequence<1, T*>& output_ptrs, 
                    const sequence<1, const T*>& input_ptrs,
                    const sequence<1, dim_list>& output_dims,
                    const sequence<1, dim_list>& input_dims);
};

template<typename T>
const char *block_permute_kernel<T>::k_clazz = "block_permute_kernel<T>";

template<typename T>
void block_permute_kernel<T>::_permute(T* output_ptr, 
                                       const T* input_ptr,
                                       const dim_list& output_dims,
                                       const dim_list& input_dims,
                                       size_t output_offset,size_t input_offset,size_t level)
{
    //Is this index permuted?
    size_t input_level = m_perm[level];

    //Determine the increment of the input array
    size_t input_incr = 1;
    if(input_level != (output_dims.size() - 1))
    {
        input_incr = std::accumulate(&input_dims[1]+input_level,&input_dims[0]+input_dims.size(),1,std::multiplies<size_t>());
    }

    if(level == (output_dims.size() - 1))
    {
        for(size_t i = 0; i < output_dims.back(); ++i)
        {
            output_ptr[output_offset + i] = input_ptr[input_offset + i*input_incr];
        }
    }
    else
    {
        size_t output_incr = std::accumulate(&output_dims[1]+level,&output_dims[0]+output_dims.size(),1,std::multiplies<size_t>());
        for(size_t i = 0; i < output_dims[level]; ++i)
        {
            _permute(output_ptr,input_ptr,output_dims,input_dims,output_offset,input_offset,level+1);

            output_offset += output_incr;
            input_offset += input_incr;

        }
    }
}


//Note: output_dims is ignored, as it is DEFINED by the permutation and the input dims
//It is kept as an argument only for interface compatibility
template<typename T>
void block_permute_kernel<T>::operator()(const sequence<1, T*>& output_ptrs, 
                                         const sequence<1, const T*>& input_ptrs,
                                         const sequence<1, dim_list>& output_dims,
                                         const sequence<1, dim_list>& input_dims)
{
	//Permutation must preserve dimensionality
    if(input_dims[0].size() != output_dims[0].size() || input_dims[0].size() != m_perm.get_order())
    {
            throw bad_parameter(g_ns, k_clazz,"operator(...)",
                    __FILE__, __LINE__, "output and input blocks do not have the same dimensionality");
    } 


    //We generate the correct output dims based on our permutation
    dim_list real_output_dims(input_dims[0]);
    for(size_t i = 0; i < input_dims[0].size(); ++i)
    {
        real_output_dims[i] = input_dims[0][m_perm[i]];
    }

    _permute(output_ptrs[0],input_ptrs[0],real_output_dims,input_dims[0]);
};


} // namespace libtensor

#endif /* BLOCK_PERMUTE_KERNEL_H */
