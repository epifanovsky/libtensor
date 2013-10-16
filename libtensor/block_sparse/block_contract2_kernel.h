#ifndef BLOCK_CONTRACT2_KERNEL_H
#define BLOCK_CONTRACT2_KERNEL_H

#include "../linalg/linalg.h"
#include "block_kernel_i.h"


//TODO: REMOVE
#include <mkl.h>
#include <iostream>

namespace libtensor {
template<typename T>

class block_contract2_kernel : public block_kernel_i<1,2,T> {
public:
    static const char *k_clazz; //!< Class name
private:
    std::vector< sequence<1,size_t> >  m_output_indices_sets; //!< Which index in each output tensor does each loop modify?
    std::vector< sequence<2,size_t> > m_input_indices_sets; //!< Which index in each input tensor does each loop modify?
    std::vector< sequence<1,bool> > m_output_ignore_sets; //!< Which output pointers are incremented by each loop?
    std::vector< sequence<2,bool> > m_input_ignore_sets; //!< Which input pointers are incremented by each loop?
    size_t m_n_loops; //!< The number of loops in this contraction

    //Determines what dgemm call to use as the base operation for this contraction

    //Ensures that the block dimensions that are passed to operator()(...) are compatible with the index set member
    //variables
    void _validate_indices(const sequence<1, dim_list>& output_dims,
                           const sequence<2, dim_list>& input_dims) const;
    
    //Recurses over the loops to perform the contraction
    void _contract_internal(const sequence<1, T*>& output_ptrs, 
                            const sequence<2, const T*>& input_ptrs,
                            const sequence<1, dim_list>& output_dims,
                            const sequence<2, dim_list>& input_dims,
                            const size_t level = 0) const;


    //Matrix multiply internals
    void (*m_dgemm_fn)(void*,size_t,size_t,size_t,const T*,size_t,const T*,size_t,T*,size_t,T); //DGEMM function to call to process deepest level
    size_t m_m;
    size_t m_n;
    size_t m_k;
    size_t m_lda;
    size_t m_ldb;
    size_t m_ldc;



public:
    //Constructor
    //May seem clunky to take a sequence<1,size_t> instead of a single value, 
    //but block_loop uses sequences, so this matches nicely with that...
    block_contract2_kernel(const std::vector< sequence<1,size_t> >&  output_indices_sets,
                           const std::vector< sequence<2,size_t> >& input_indices_sets,
                           const std::vector< sequence<1,bool> >& output_ignore,
                           const std::vector< sequence<2,bool> >& input_ignore);

    void operator()(const sequence<1, T*>& output_ptrs, 
                    const sequence<2, const T*>& input_ptrs,
                    const sequence<1, dim_list>& output_dims,
                    const sequence<2, dim_list>& input_dims);
};

template<typename T>
const char *block_contract2_kernel<T>::k_clazz = "block_contract2_kernel<T>";


//Ensures that the block dimensions that are passed to operator()(...) are compatible with the index set member
//variables
template<typename T>
void block_contract2_kernel<T>::_validate_indices(const sequence<1, dim_list>& output_dims,
                                                  const sequence<2, dim_list>& input_dims) const
{
    
    //Make sure that the dimensions match for all loops for all tensors
    //and that none of the requested indices are out of bounds
    for(size_t i = 0; i < m_output_indices_sets.size(); ++i)
    {
        const sequence<1,size_t>& cur_output_indices = m_output_indices_sets[i];
        const sequence<2,size_t>& cur_input_indices = m_input_indices_sets[i];

        //Check that no indices are out of bounds
        if(cur_output_indices[0] > (output_dims[0].size() - 1)  )
        {
            throw bad_parameter(g_ns, k_clazz,"_validate_indices(...)",
                    __FILE__, __LINE__, "Output index out of bounds!");
        }
        if((cur_input_indices[0] > (input_dims[0].size() - 1)) || (cur_input_indices[1] > (input_dims[1].size() - 1)))
        {
            throw bad_parameter(g_ns, k_clazz,"_validate_indices(...)",
                    __FILE__, __LINE__, "Input index out of bounds!");
        }
        
        //Check that all sizes are compatible
        std::vector<size_t> cur_dim_vec;
        if(!m_output_ignore_sets[i][0])
        {
            cur_dim_vec.push_back(output_dims[0][cur_output_indices[0]]);
        }
        if(!m_input_ignore_sets[i][0])
        {
            cur_dim_vec.push_back(input_dims[0][cur_input_indices[0]]);
        }
        if(!m_input_ignore_sets[i][1])
        {
            cur_dim_vec.push_back(input_dims[1][cur_input_indices[1]]);
        }

        for(size_t j = 0; j < cur_dim_vec.size(); ++j)
        {
            if(cur_dim_vec[j] != cur_dim_vec[0])
            {
                throw bad_parameter(g_ns, k_clazz,"_validate_indices(...)",
                        __FILE__, __LINE__, "index dimensions are not compatible");
            }
        }
    }
}

//Recurses over the loops to perform the contraction
template<typename T>
void block_contract2_kernel<T>::_contract_internal(const sequence<1, T*>& output_ptrs, 
                                                   const sequence<2, const T*>& input_ptrs,
                                                   const sequence<1, dim_list>& output_dims,
                                                   const sequence<2, dim_list>& input_dims,
                                                   const size_t level) const
{
    //Base case: call matmul kernel
    if(level == m_n_loops - 3)
    {
        (*m_dgemm_fn)(NULL,m_m,m_n,m_k,input_ptrs[0],m_lda,input_ptrs[1],m_ldb,output_ptrs[0],m_ldc,1.0);
    }
    else
    {

        //Compute the stride for each tensor
        size_t output_stride = 1;
        if(!m_output_ignore_sets[level][0])
        {
            for(size_t i = m_output_indices_sets[level][0]+1; i < output_dims[0].size(); ++i)
            {
                output_stride *= output_dims[0][i];
            }
        }

        //TODO: Need to account for ignores in here....

        size_t input_stride_1 = 1;
        if(!m_input_ignore_sets[level][0])
        {
            for(size_t i = m_input_indices_sets[level][0]+1; i < input_dims[0].size(); ++i)
            {
                input_stride_1 *= input_dims[0][i];
            }
        }

        size_t input_stride_2 = 1;
        if(!m_input_ignore_sets[level][1])
        {
            for(size_t i = m_input_indices_sets[level][1]+1; i < input_dims[1].size(); ++i)
            {
                input_stride_2 *= input_dims[1][i];
            }
        }

        sequence<1,T*> new_output_ptrs(output_ptrs);
        sequence<2,const T*> new_input_ptrs(input_ptrs);
        for(size_t i = 0; i < output_dims[0][m_output_indices_sets[level][0]]; ++i)
        {
            _contract_internal(new_output_ptrs,new_input_ptrs,output_dims,input_dims,level+1);
            if(!m_output_ignore_sets[level][0])
            {
                new_output_ptrs[0] += output_stride; 
            }
            if(!m_input_ignore_sets[level][0])
            {
                new_input_ptrs[0] += input_stride_1;
            }
            if(!m_input_ignore_sets[level][1])
            {
                new_input_ptrs[1] += input_stride_2;
            }
        }
    }
}

//Constructor
template<typename T>
block_contract2_kernel<T>::block_contract2_kernel(const std::vector< sequence<1,size_t> >&  output_indices_sets,
                                                  const std::vector< sequence<2,size_t> >& input_indices_sets,
                                                  const std::vector< sequence<1,bool> >& output_ignore,
                                                  const std::vector< sequence<2,bool> >& input_ignore) : m_output_indices_sets(output_indices_sets),
                                                                                                         m_input_indices_sets(input_indices_sets),
                                                                                                         m_output_ignore_sets(output_ignore),
                                                                                                         m_input_ignore_sets(input_ignore)
{
    //Validate that all of the vectors have matching lengths
    //Save that common length as the number of loops
    m_n_loops = output_indices_sets.size();
    if(m_n_loops != input_indices_sets.size() || m_n_loops != output_ignore.size() || m_n_loops != input_ignore.size())
    {
        throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
                __FILE__, __LINE__, "Sizes of arguments do not all match!");
    } 

}

template<typename T>
void block_contract2_kernel<T>::operator()(const sequence<1, T*>& output_ptrs, 
                                           const sequence<2, const T*>& input_ptrs,
                                           const sequence<1, dim_list>& output_dims,
                                           const sequence<2, dim_list>& input_dims)
{
    //Ensure that the dimensions we were passed are valid
    //Too high order?
    if((output_dims[0].size() > m_n_loops) || (input_dims[0].size() > m_n_loops) || (input_dims[1].size() > m_n_loops))
    {
        throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                __FILE__, __LINE__, "Size of dimension list is too big");
    }

    //TODO: This catches lots of other exceptions...some of them may be redundant
    //TODO: Could move this into constructor, compare indices_sets entries to their neighbors, rather than to dim
    //vectors...would also catch invalid cases, possibly save expense
    //Is the last loop over an index that exists in the output tensor not the last index in the output?
    //If so, the output is strided
    for(size_t i = m_n_loops - 1; i >= 0; --i)
    {
        if(m_output_ignore_sets[i][0])
        {
            continue;
        }
        if(m_output_indices_sets[i][0] != (output_dims[0].size() - 1))
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "Strided output is not supported");
        }
        else
        {
            break;
        }
    }

    //Determine what dgemm routine to call for the innermost matrix multiply 
    //TODO: Put some of these array references in variables!!!!
    //Determine the position of the i loop and j loop in order to determine which kernel to call
    if(m_input_indices_sets[m_n_loops - 3][0] == (input_dims[0].size() - 2)) //A no transpose
    {
        if(m_input_indices_sets[m_n_loops - 2][1] == (input_dims[1].size() - 1)) //B no transpose
        {
            m_dgemm_fn = &linalg::mul2_ij_ip_pj_x;
            m_m = input_dims[0][input_dims[0].size() - 2];
            m_n = input_dims[1][input_dims[1].size() - 1];
            m_k = input_dims[1][input_dims[1].size() - 2];
            m_lda = m_k;
            m_ldb = m_n;
            m_ldc = m_n;
        }
        else if(m_input_indices_sets[m_n_loops - 2][1] == (input_dims[1].size() - 2)) //B transpose
        {
            m_dgemm_fn = &linalg::mul2_ij_ip_jp_x;
            m_m = input_dims[0][input_dims[0].size() - 2];
            m_n = input_dims[1][input_dims[1].size() - 2];
            m_k = input_dims[1][input_dims[1].size() - 1];
            m_lda = m_k;
            m_ldb = m_k;
            m_ldc = m_n;
        }
        else
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "Non-matrix multiply isomorphic kernel is not supported");
        }
    }
    else if(m_input_indices_sets[m_n_loops - 3][0] == (input_dims[0].size() - 1)) // A transpose
    {
        if(m_input_indices_sets[m_n_loops - 2][1] == (input_dims[1].size() - 1)) //B no transpose
        {
            m_dgemm_fn = &linalg::mul2_ij_pi_pj_x;
            m_m = input_dims[0][input_dims[0].size() - 1];
            m_n = input_dims[1][input_dims[1].size() - 1];
            m_k = input_dims[1][input_dims[1].size() - 2];
            m_lda = m_m;
            m_ldb = m_n;
            m_ldc = m_n;
        }
        else if(m_input_indices_sets[m_n_loops - 2][1] == (input_dims[1].size() - 2)) //B transpose
        {
            m_dgemm_fn = &linalg::mul2_ij_pi_jp_x;
            m_m = input_dims[0][input_dims[0].size() - 1];
            m_n = input_dims[1][input_dims[1].size() - 2];
            m_k = input_dims[1][input_dims[1].size() - 1];
            m_lda = m_m;
            m_ldb = m_k;
            m_ldc = m_n;
        }
        else
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "Non-matrix multiply isomorphic kernel is not supported");
        }
    }
    else
    {
        throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                __FILE__, __LINE__, "Non-matrix multiply isomorphic kernel is not supported");
    }

    _validate_indices(output_dims,input_dims);
    _contract_internal(output_ptrs,input_ptrs,output_dims,input_dims);
}


} // namespace libtensor


#endif /* BLOCK_CONTRACT2_KERNEL_H */
