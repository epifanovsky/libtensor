#ifndef BLOCK_PERMUTE_KERNEL_H
#define BLOCK_PERMUTE_KERNEL_H

#include <map>
#include <numeric>
#include "block_kernel_i.h"

namespace libtensor { 

typedef std::map<size_t,size_t> permute_map;  

//!!!!! MAPS ARE PASSED IN WITH THE CONVENTION input->output!
template<typename T>
class block_permute_kernel : public block_kernel_i<1,1,T> {
public:
    static const char *k_clazz; //!< Class name
private:
    permute_map m_perm;

    //Recurse internal permutation handler
    void _permute(T* output_ptrs, 
                  T* input_ptrs,
                  const dim_list& output_dims,
                  const dim_list& input_dims,
                  size_t output_offset = 0,size_t input_offset = 0,size_t level = 0);
public:
    block_permute_kernel(permute_map& perm);

    //Returns a pointer to a copy of this object
    block_kernel_i<1,1,T>* clone() const { return (block_kernel_i<1,1,T>*) new block_permute_kernel(*this); };  

    void operator()(const sequence<1, T*>& output_ptrs, 
                    const sequence<1, T*>& input_ptrs,
                    const sequence<1, dim_list>& output_dims,
                    const sequence<1, dim_list>& input_dims);

};

template<typename T>
const char *block_permute_kernel<T>::k_clazz = "block_permute_kernel<T>";

//Validates the requested permutation
template<typename T>
block_permute_kernel<T>::block_permute_kernel(permute_map& perm)
{
    //Pre-process the map for application later
    for(permute_map::iterator pm_it = perm.begin(); pm_it != perm.end(); ++pm_it)
    {
        //Verify that the permutation is complete (that all permuted indices are assigned to positions
std:
        if(perm.find(pm_it->second) == perm.end())
        {
            throw bad_parameter(g_ns, k_clazz,"block_permute_kernel(...)",
                    __FILE__, __LINE__, "Incomplete permutation map was passed to constructor");
        }

        //Now invert the map, as the representation of output->input is more convenient to work in
        //Cannot map two indices to the same location
        if(m_perm.find(pm_it->second) != m_perm.end()) 
        {
            throw bad_parameter(g_ns, k_clazz,"block_permute_kernel(...)",
                    __FILE__, __LINE__, "Cannot map two indices to the same location");
        }
        m_perm[pm_it->second] = pm_it->first;
    }
}

template<typename T>
void block_permute_kernel<T>::_permute(T* output_ptr, 
                                         T* input_ptr,
                                         const dim_list& output_dims,
                                         const dim_list& input_dims,
                                         size_t output_offset,size_t input_offset,size_t level)
{
    //Is this index permuted?
    size_t input_level; 
    if(m_perm.find(level) != m_perm.end())
    {
        input_level = m_perm[level];
    }
    else
    {
        input_level = level;
    }

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
                                         const sequence<1, T*>& input_ptrs,
                                         const sequence<1, dim_list>& output_dims,
                                         const sequence<1, dim_list>& input_dims)
{


    if(input_dims[0].size() != output_dims[0].size())
    {
            throw bad_parameter(g_ns, k_clazz,"operator(...)",
                    __FILE__, __LINE__, "output and input blocks do not have the same dimensionality");
    } 

    //Check that our map is valid for the block dimensions that we have been passed
    for(permute_map::iterator pm_it = m_perm.begin(); pm_it != m_perm.end(); ++pm_it)
    {
        if((pm_it->first > (output_dims[0].size() - 1)) || (pm_it->second > (output_dims[0].size() - 1)))
        {
            throw bad_parameter(g_ns, k_clazz,"operator(...)",
                    __FILE__, __LINE__, "Permutation map exceeds bounds of block dimensions");
        }
    } 

    //We generate the correct output dims based on our permutation
    //Remember, our map is inverted: now is output -> input
    dim_list real_output_dims(input_dims[0]);
    for(permute_map::iterator pm_it = m_perm.begin(); pm_it != m_perm.end(); ++pm_it)
    {
        real_output_dims[pm_it->first] = input_dims[0][pm_it->second];
    } 

    _permute(output_ptrs[0],input_ptrs[0],real_output_dims,input_dims[0]);
};


} // namespace libtensor

#endif /* BLOCK_PERMUTE_KERNEL_H */
