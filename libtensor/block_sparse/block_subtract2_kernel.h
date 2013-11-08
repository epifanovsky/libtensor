#ifndef BLOCK_SUBTRACT2_KERNEL_H
#define BLOCK_SUBTRACT2_KERNEL_H

#include <vector>
#include "block_kernel_i.h"

namespace libtensor {

template<typename T>
class block_subtract2_kernel : public block_kernel_i<1,2,T> {
public:
    static const char *k_clazz; //!< Class name
private:
    std::vector< sequence<1,size_t> >  m_output_indices_sets; //!< Which index in each output tensor does each loop modify?
    std::vector< sequence<2,size_t> > m_input_indices_sets; //!< Which index in each input tensor does each loop modify?
    std::vector< sequence<1,bool> > m_output_ignore_sets; //!< Which output pointers are incremented by each loop?
    std::vector< sequence<2,bool> > m_input_ignore_sets; //!< Which input pointers are incremented by each loop?

public:
    //Constructor
    //May seem clunky to take a sequence<1,size_t> instead of a single value, 
    //but block_loop uses sequences, so this matches nicely with that...
    block_subtract2_kernel(const std::vector< sequence<1,size_t> >&  output_indices_sets,
                           const std::vector< sequence<2,size_t> >& input_indices_sets,
                           const std::vector< sequence<1,bool> >& output_ignore,
                           const std::vector< sequence<2,bool> >& input_ignore);

    void operator()(const sequence<1, T*>& output_ptrs, 
                    const sequence<2, const T*>& input_ptrs,
                    const sequence<1, dim_list>& output_dims,
                    const sequence<2, dim_list>& input_dims);


};

template<typename T>
const char *block_subtract2_kernel<T>::k_clazz = "block_subtract2_kernel<T>";

//Constructor
template<typename T>
block_subtract2_kernel<T>::block_subtract2_kernel(const std::vector< sequence<1,size_t> >&  output_indices_sets,
                                                  const std::vector< sequence<2,size_t> >& input_indices_sets,
                                                  const std::vector< sequence<1,bool> >& output_ignore,
                                                  const std::vector< sequence<2,bool> >& input_ignore) : m_output_indices_sets(output_indices_sets),
                                                                                                         m_input_indices_sets(input_indices_sets),
                                                                                                         m_output_ignore_sets(output_ignore),
                                                                                                         m_input_ignore_sets(input_ignore)
{
}

//Constructor
template<typename T>
void block_subtract2_kernel<T>::operator()(const sequence<1, T*>& output_ptrs, 
                                           const sequence<2, const T*>& input_ptrs,
                                           const sequence<1, dim_list>& output_dims,
                                           const sequence<2, dim_list>& input_dims)
{
    //TODO Support permuted block orders - for now, just assume all block sizes equal
    size_t total_block_size = 1;
    for(size_t i = 0; i < output_dims[0].size(); ++i)
    {
        total_block_size *= output_dims[0][i]; 
    }

    for(size_t i = 0; i < total_block_size; ++i)
    {
        output_ptrs[0][i] = input_ptrs[0][i] - input_ptrs[1][i];
    }
}

} // namespace libtensor 


#endif /* BLOCK_SUBTRACT2_KERNEL_H */
