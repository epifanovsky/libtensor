#ifndef BLOCK_CONTRACT2_KERNEL_H
#define BLOCK_CONTRACT2_KERNEL_H

#include <algorithm>
#include "../linalg/linalg.h"
#include "block_kernel_i.h"

//TODO REMOVE
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
                            const size_t n_contracted_inds,
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
                                                   const size_t n_contracted_inds,
                                                   const size_t level) const
{
    //Base case: call matmul kernel
    if(level == m_n_loops - n_contracted_inds - 2)
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
            _contract_internal(new_output_ptrs,new_input_ptrs,output_dims,input_dims,n_contracted_inds,level+1);
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
    
    //Asssemble list of contracted indices in A, and the last index in C that also appears in A
    //All indices that are ignored by output tensor are contracted
    size_t M = input_dims[0].size();
    size_t A_inds_end_in_C; 
    std::vector<size_t> A_contracted_inds;
    size_t last_A_idx_in_C;
    for(size_t i = 0; i < m_n_loops; ++i)
    {
        if(m_output_ignore_sets[i][0])
        {
            A_contracted_inds.push_back(m_input_indices_sets[i][0]);
        }
        else if(!m_input_ignore_sets[i][0])
        {
            //Not ignored in output, so must be in C as well
            last_A_idx_in_C = m_input_indices_sets[i][0];
        }
    }

    //Do the same for B
    //Appropriate index from B must immediately follow A_inds_end_in_C
    size_t N = input_dims[1].size();
    std::vector<size_t> B_contracted_inds;
    size_t first_B_idx_in_C;
    for(size_t i = 0; i < m_n_loops; ++i)
    {
        if(m_output_ignore_sets[i][0])
        {
            B_contracted_inds.push_back(m_input_indices_sets[i][1]);
        }
        else if(!m_input_ignore_sets[i][1])
        {
            first_B_idx_in_C = m_input_indices_sets[i][1];
        }
    }

    //Number of contracted indices must be the same in A and B
    if(A_contracted_inds.size() != B_contracted_inds.size())
    {
        throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                __FILE__, __LINE__, "number of contracted indices must be the same");
    }
    size_t n_contracted_inds = A_contracted_inds.size();

    //Determine if A is transposed. It is transposed if the contracted indices appear in reverse order at the beginning
    //of A
    bool A_transposed;
    bool invalid = false;
    if(A_contracted_inds[n_contracted_inds- 1] == 0)
    {
        //If A is transposed, then ALL contracted indices must be grouped together at the BEGINNING
        A_transposed = true;
        for(size_t i = 0; i < n_contracted_inds; ++i)
        {
            if(A_contracted_inds[i] != i)
            {
                invalid = true;
                break;
            }
        }
        //Additionally, the FIRST non-contracted index must be the last index from A appearing in C 
        //in order to use a matmul
        if(last_A_idx_in_C != n_contracted_inds)
        {
            invalid = true;
        }
    }
    else
    {
        //If A is not transposed, then ALL contracted indices must be grouped together in order at the END
        A_transposed = false;
        for(size_t i = 0; i < n_contracted_inds; ++i)
        {
            if(A_contracted_inds[i] != M - n_contracted_inds + i)
            {
                invalid = true;
                break;
            }
        }
        //Addtionally, the LAST non-contracted index must be the last index from A appearing in C
        if(last_A_idx_in_C != M - n_contracted_inds - 1)
        {
            invalid = true;
        }
    }
    if(invalid)
    {
        throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                __FILE__, __LINE__, "Non-matrix multiply isomorphic kernel is not supported");
    }

    //Determine if B is transposed - rules are opposite those of A
    bool B_transposed;
    if(B_contracted_inds[0] == 0)
    {
        //If B is not transposed, then ALL contracted indices must be grouped together in order at the BEGINNING
        B_transposed = false;
        for(size_t i = 0; i < n_contracted_inds; ++i)
        {
            if(B_contracted_inds[i] != i)
            {
                invalid = true;
                break;
            }
        }
        //Additionally, the FIRST non-contracted index must be the first index from B appearing in C
        if(first_B_idx_in_C != n_contracted_inds)
        {
            invalid = true;
        }
    }
    else
    {
        //If B is transposed, then ALL contracted indices must be grouped together in order at the END
        B_transposed = true;
        for(size_t i = 0; i < n_contracted_inds; ++i)
        {
            if(B_contracted_inds[i] != N - n_contracted_inds + i)
            {
                invalid = true;
                break;
            }
        }
        //Additionally, the LAST non-contracted index must be the FIRST index from B appearing in C
        if(first_B_idx_in_C != N - n_contracted_inds - 1)
        {
            invalid = true;
        }
    }
    if(invalid)
    {
        throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                __FILE__, __LINE__, "Non-matrix multiply isomorphic kernel is not supported");
    }

    if(!A_transposed)
    {
        m_m = input_dims[0][M - n_contracted_inds - 1];
        m_k = 1;
        for(size_t contr = 0; contr < n_contracted_inds; ++contr)
        {
            m_k *= input_dims[0][M - n_contracted_inds + contr];
        }

        if(!B_transposed)
        {
            m_dgemm_fn = &linalg::mul2_ij_ip_pj_x;
            m_n = input_dims[1][n_contracted_inds];
            m_lda = m_k;
            m_ldb = m_n;
            m_ldc = m_n;
        }
        else
        {
            m_dgemm_fn = &linalg::mul2_ij_ip_jp_x;
            m_n = input_dims[1][0];
            m_lda = m_k;
            m_ldb = m_k;
            m_ldc = m_n;
        }
    }
    else
    {
        m_m = input_dims[0][n_contracted_inds];
        m_k = 1;
        for(size_t contr = 0; contr < n_contracted_inds; ++contr)
        {
            m_k *= input_dims[0][contr];
        }

        if(!B_transposed)
        {
            m_dgemm_fn = &linalg::mul2_ij_pi_pj_x;
            m_n = input_dims[1][n_contracted_inds];
            m_lda = m_m;
            m_ldb = m_n;
            m_ldc = m_n;
        }
        else
        {
            m_dgemm_fn = &linalg::mul2_ij_pi_jp_x;
            m_n = input_dims[1][n_contracted_inds - 1];
            m_lda = m_m;
            m_ldb = m_k;
            m_ldc = m_n;
        }
    }

    _validate_indices(output_dims,input_dims);
    _contract_internal(output_ptrs,input_ptrs,output_dims,input_dims,n_contracted_inds);
}


} // namespace libtensor


#endif /* BLOCK_CONTRACT2_KERNEL_H */
