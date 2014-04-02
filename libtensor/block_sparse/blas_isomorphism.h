#ifndef BLAS_ISOMORPHISM_H
#define BLAS_ISOMORPHISM_H

#include "sparse_loop_list.h"
#include "../linalg/linalg.h"

namespace libtensor {

template<typename T>
class matmul_isomorphism_params
{
public:
    static const char* k_clazz; //!< Class name
    typedef void (*dgemm_fp_t)(void*,size_t,size_t,size_t,const T*,size_t,const T*,size_t,T*,size_t,T); //DGEMM function to call to process deepest level
private:
    runtime_permutation m_A_perm;
    runtime_permutation m_B_perm;

    dgemm_fp_t m_dgemm_fp;
    static const dgemm_fp_t dgemm_fp_arr[4];
                
public:
    matmul_isomorphism_params(const sparse_loop_list& sll);
    runtime_permutation get_perm_A() { return m_A_perm; }
    runtime_permutation get_perm_B() { return m_B_perm; }
    dgemm_fp_t get_dgemm_fp() { return m_dgemm_fp; }
};

template<typename T>
const char* matmul_isomorphism_params<T>::k_clazz = "matmul_isomorphism_params<T>";

template<typename T>
const typename matmul_isomorphism_params<T>::dgemm_fp_t matmul_isomorphism_params<T>::dgemm_fp_arr[4] = {&linalg::mul2_ij_ip_pj_x,
                                                                                                         &linalg::mul2_ij_ip_jp_x,
                                                                                                         &linalg::mul2_ij_pi_pj_x,
                                                                                                         &linalg::mul2_ij_pi_jp_x};

//Determine the type of matmul required to execute the contraction:
//	A.B     [ C_ij = A_ik B_kj ]
//  A.B^T   [ C_ij = A_ik B_jk ]
//	A^T.B   [ C_ij = A_ki B_kj ]
//  A^T.B^T [ C_ij = A_ki B_jk ]
//
//  If the contraction cannot be executed in this fashion, return the permutations 
//  necessary to make it happen
template<typename T>
matmul_isomorphism_params<T>::matmul_isomorphism_params(const sparse_loop_list& sll) : m_A_perm(0),m_B_perm(0) 
{
    std::vector<sparse_bispace_any_order> bispaces = sll.get_bispaces();
    if(bispaces.size() != 3)
    {
        throw bad_parameter(g_ns, k_clazz,"matmul_isomorphism_params(...)",__FILE__, __LINE__, "Only 3-bispace contractions can be handled via matmul at this time");
    }
    
    //First we find all the contracted indices    
	//Figure out what RHS subspace indices are contracted, based on whether
	//the loops that traverse them don't touch the output tensor
	std::vector<size_t> A_contracted_indices;
	std::vector<size_t> B_contracted_indices;
    std::vector<block_loop> loops = sll.get_loops();
	for(size_t loop_idx = 0; loop_idx < loops.size(); ++loop_idx)
	{
		//Contracted? It should be if it doesn't appear in the output tensor
		const block_loop& cur_loop = loops[loop_idx];
		if(cur_loop.is_bispace_ignored(0))
		{
			//If contracted idx doesn't appear in both input tensors, it shouldn't be contracted
			if((cur_loop.is_bispace_ignored(1)) || (cur_loop.is_bispace_ignored(2)))
			{
				throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",__FILE__, __LINE__, 
                        "an idx not present in both A and B should appear in C");
			}
			A_contracted_indices.push_back(cur_loop.get_subspace_looped(1));
			B_contracted_indices.push_back(cur_loop.get_subspace_looped(2));
		}
	}
    if(A_contracted_indices.size() != B_contracted_indices.size())
    {
				throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",__FILE__, __LINE__, 
                        "an idx not present in both A and B should appear in C");

    }
    size_t n_contr = A_contracted_indices.size();
    sort(A_contracted_indices.begin(),A_contracted_indices.end());
    sort(B_contracted_indices.begin(),B_contracted_indices.end());

    //Figure out the permutation necessary to move A and B to Aik Bjk contraction format
    bool A_trans = false;
    bool B_trans = true;
	size_t A_order = bispaces[1].get_order();
	size_t B_order = bispaces[2].get_order();
    m_A_perm = runtime_permutation(A_order);
    m_B_perm = runtime_permutation(B_order);
    for(size_t A_dest_idx = A_order - n_contr; A_dest_idx < A_order; ++A_dest_idx)
    {
        size_t A_src_idx = A_contracted_indices[A_dest_idx - (A_order - n_contr)];
        m_A_perm.permute(A_src_idx,A_dest_idx);
    }
    for(size_t B_dest_idx = B_order - B_contracted_indices.size(); B_dest_idx < B_order; ++B_dest_idx)
    {
        size_t B_src_idx = B_contracted_indices[B_dest_idx - (B_order - n_contr)];
        m_B_perm.permute(B_src_idx,B_dest_idx);
    }

    //Check if we need to reverse transposition option
    //This is the case if A perm = {...0,1,2,END} and same for B_perm
    runtime_permutation A_end_entries(std::vector<size_t>(m_A_perm.end() - n_contr,m_A_perm.end()));
    runtime_permutation B_end_entries(std::vector<size_t>(m_B_perm.end() - n_contr,m_B_perm.end()));
    if(A_end_entries == runtime_permutation(n_contr))
    {
        A_trans = !A_trans;
        m_A_perm = runtime_permutation(A_order);
    }
    if(B_end_entries == runtime_permutation(n_contr))
    {
        B_trans = !B_trans;
        m_B_perm = runtime_permutation(B_order);
    }

    m_dgemm_fp = dgemm_fp_arr[(size_t)A_trans*2+(size_t)B_trans];
}

} // namespace libtensor

#endif /* BLAS_ISOMORPHISM_H */
