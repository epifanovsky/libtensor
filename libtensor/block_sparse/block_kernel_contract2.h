#ifndef BLOCK_KERNEL_CONTRACT2_H
#define BLOCK_KERNEL_CONTRACT2_H

#include "block_kernel_i.h"
#include "block_kernel_permute.h"
#include "matmul_isomorphism_params.h"
#include "../linalg/linalg.h"

namespace libtensor
{

template<typename T>
class block_kernel_contract2: public libtensor::block_kernel_i<block_kernel_contract2<T>,T>
{
private:
    static const char* k_clazz; //!< Class name
    std::vector<size_t> m_block_orders; //!< Orders of each of the blocks

    typedef void (*dgemm_fp_t)(void*,size_t,size_t,size_t,const T*,size_t,const T*,size_t,T*,size_t,T); //DGEMM function to call to process deepest level
    dgemm_fp_t m_dgemm_fp;
    static const dgemm_fp_t dgemm_fp_arr[4];

    std::vector<idx_pair_list> m_ts_groups;
    size_t m_n_contr_inds;

    void (*m_dgemv_fn)(void*,size_t,size_t,const T*,size_t,const T*,size_t,T*,size_t,T);

    //Sets everything up if we are doing a matrix-vector multiply
    void init_matvec(const std::vector<sparse_bispace_impl>& bispaces,
                     const idx_list& A_contr_inds,
                     const idx_list& B_contr_inds);

    //Is the vector the first argument in the matrix vector multiply contraction?
    size_t m_vec_bispace_idx;
    size_t m_mat_bispace_idx;

    //matmul data
    bool m_A_trans;
    bool m_B_trans;
    std::vector<runtime_permutation> m_perms;
    block_kernel_permute<T> m_C_perm_kern;
    std::vector<block_kernel_permute<T> > m_perm_kerns;
    std::vector<runtime_permutation> m_ident_perms;
    std::vector<dim_list> m_perm_dim_lists;
    std::vector<T*> m_perm_ptrs;
    dim_list m_perm_ptr_sizes;
public:
    block_kernel_contract2(const std::vector<sparse_bispace_impl>& bispaces,
                           const std::vector<idx_pair_list>& ts_groups);
    void operator()(const std::vector<T*>& ptrs, const std::vector<dim_list>& dim_lists);

    block_kernel_contract2(const block_kernel_contract2<T>& rhs);
    block_kernel_contract2<T>& operator=(const block_kernel_contract2<T>& rhs);
    ~block_kernel_contract2();
};

template<typename T>
const typename block_kernel_contract2<T>::dgemm_fp_t block_kernel_contract2<T>::dgemm_fp_arr[4] = {&linalg::mul2_ij_ip_pj_x,
                                                                                                   &linalg::mul2_ij_ip_jp_x,
                                                                                                   &linalg::mul2_ij_pi_pj_x,
                                                                                                   &linalg::mul2_ij_pi_jp_x};

template<typename T>
const char* block_kernel_contract2<T>::k_clazz = "block_kernel_contract2<T>";

} /* namespace libtensor */

template<typename T>
libtensor::block_kernel_contract2<T>::block_kernel_contract2(const block_kernel_contract2<T>& rhs) : m_block_orders(rhs.m_block_orders),
                                                                                                     m_dgemm_fp(rhs.m_dgemm_fp),
                                                                                                     m_ts_groups(rhs.m_ts_groups),
                                                                                                     m_n_contr_inds(rhs.m_n_contr_inds),
                                                                                                     m_vec_bispace_idx(rhs.m_vec_bispace_idx),
                                                                                                     m_mat_bispace_idx(rhs.m_mat_bispace_idx),
                                                                                                     m_A_trans(rhs.m_A_trans),
                                                                                                     m_B_trans(rhs.m_B_trans),
                                                                                                     m_perms(rhs.m_perms),
                                                                                                     m_C_perm_kern(rhs.m_C_perm_kern),
                                                                                                     m_perm_kerns(rhs.m_perm_kerns),
                                                                                                     m_ident_perms(rhs.m_ident_perms),
                                                                                                     m_perm_dim_lists(rhs.m_perm_dim_lists),
                                                                                                     m_perm_ptr_sizes(rhs.m_perm_ptr_sizes),
                                                                                                     m_perm_ptrs(3)
{
    for(size_t perm_idx = 0; perm_idx < m_perms.size(); ++perm_idx)
    {
        if(m_perms[perm_idx] != m_ident_perms[perm_idx])
        {
            m_perm_ptrs[perm_idx] = new T[m_perm_ptr_sizes[perm_idx]];
        }
    }
}

template<typename T>
libtensor::block_kernel_contract2<T>& libtensor::block_kernel_contract2<T>::operator=(const block_kernel_contract2<T>& rhs)
{
    m_block_orders = rhs.m_block_orders;
    m_dgemm_fp = rhs.m_dgemm_fp;
    m_ts_groups = rhs.m_ts_groups;
    m_n_contr_inds = rhs.m_n_contr_inds;
    m_vec_bispace_idx = rhs.m_vec_bispace_idx;
    m_mat_bispace_idx = rhs.m_mat_bispace_idx;
    m_A_trans = rhs.m_A_trans;
    m_B_trans = rhs.m_B_trans;
    m_perms = rhs.m_perms;
    m_C_perm_kern = rhs.m_C_perm_kern;
    m_perm_kerns = rhs.m_perm_kerns;
    m_ident_perms = rhs.m_ident_perms;
    m_perm_dim_lists = rhs.m_perm_dim_lists;
    m_perm_ptr_sizes = rhs.m_perm_ptr_sizes;
    m_perm_ptrs.resize(3,NULL);
    for(size_t perm_idx = 0; perm_idx < m_perms.size(); ++perm_idx)
    {
        if(m_perms[perm_idx] != m_ident_perms[perm_idx])
        {
            m_perm_ptrs[perm_idx] = new T[m_perm_ptr_sizes[perm_idx]];
        }
    }
}

template<typename T>
libtensor::block_kernel_contract2<T>::~block_kernel_contract2()
{
    for(size_t perm_idx = 0; perm_idx < m_perms.size(); ++perm_idx)
    {
        if(m_perms[perm_idx] != m_ident_perms[perm_idx])
        {
            delete [] m_perm_ptrs[perm_idx];
        }
    }
}
template<typename T>
libtensor::block_kernel_contract2<T>::block_kernel_contract2(const std::vector<sparse_bispace_impl>& bispaces,
                                                             const std::vector<idx_pair_list>& ts_groups) : 
                                       m_ts_groups(ts_groups),
                                       m_dgemm_fp(NULL),
                                       m_perm_dim_lists(3),
                                       m_perms(3,runtime_permutation(0)),
                                       m_C_perm_kern(runtime_permutation(0)),
                                       m_ident_perms(3,runtime_permutation(0)),
                                       m_perm_ptrs(3),
                                       m_perm_ptr_sizes(3),
                                       m_perm_kerns(3,block_kernel_permute<T>(runtime_permutation(0)))
{
	//Ensure that we have a valid number of bispaces
	if(bispaces.size() != 3)
	{
		throw bad_parameter(g_ns, k_clazz,"block_kernel_contract2(...)",__FILE__, __LINE__, "sparse_loop_list must contain at least 3 bispaces");
	}

	//Simplest contraction is matrix vector multiply and requires 2 loops
    if(m_ts_groups.size() < 2)
    {
        throw bad_parameter(g_ns, k_clazz,"block_kernel_contract2(...)",__FILE__, __LINE__, "must be at least two ts_groups (indices)");
    }

	//Record the orders of the blocks that this contract kernel takes as input when run so that we can check it at
    //runtime
	for(size_t block_idx = 0; block_idx < bispaces.size(); ++block_idx)
	{
		m_block_orders.push_back(bispaces[block_idx].get_order());
	}

	//Figure out what RHS subspace indices are contracted, based on whether
	//the loops that traverse them don't touch the output tensor
	idx_list A_contr_inds;
	idx_list B_contr_inds;
    for(size_t grp_idx = 0; grp_idx < ts_groups.size(); ++grp_idx)
    {
        const idx_pair_list& grp = ts_groups[grp_idx];
        bool touches_0 = false;
        bool touches_1 = false;
        bool touches_2 = false;
        for(size_t ent_idx = 0; ent_idx < grp.size(); ++ent_idx)
        {
            if(grp[ent_idx].first == 0)
                touches_0 = true;
            if(grp[ent_idx].first == 1)
            {
                if(!touches_0)
                    A_contr_inds.push_back(grp[ent_idx].second);
                touches_1 = true;
            }
            else if(grp[ent_idx].first == 2)
            {
                if(!touches_0)
                    B_contr_inds.push_back(grp[ent_idx].second);
                touches_2 = true;
            }
        }
        if(!touches_0 && (!touches_1 || !touches_2))
        {
            throw bad_parameter(g_ns, k_clazz,"block_kernel_contract2(...)",__FILE__, __LINE__, "an idx not present in both A and B should appear in C");
        }
        if(touches_0 && !(touches_1 || touches_2))
        {
            throw bad_parameter(g_ns, k_clazz,"block_kernel_contract2(...)",__FILE__, __LINE__, "an idx present in C must appear in A or B");
        }
    }
	m_n_contr_inds = A_contr_inds.size();


	if(m_n_contr_inds == 0)
	{
		throw bad_parameter(g_ns, k_clazz,"block_kernel_contract2(...)",
				__FILE__, __LINE__, "must have at least one contracted index");
	}

    //First out if we are doing a matrix-matrix multiply or a matrix-vector multiply
    if((A_contr_inds.size() == ts_groups.size()) || (B_contr_inds.size() == ts_groups.size()))
    {
        //No dot products
		throw bad_parameter(g_ns, k_clazz,"block_kernel_contract2(...)",
				__FILE__, __LINE__, "All loops cannot correspond to contracted indices");
    }
    else if((A_contr_inds.size() == bispaces[1].get_order()) || (B_contr_inds.size() == bispaces[2].get_order()))
    {
        init_matvec(bispaces,A_contr_inds,B_contr_inds);
    }
    else
    {
        matmul_isomorphism_params mip(bispaces,ts_groups);
        m_A_trans = mip.get_A_trans();
        m_B_trans = mip.get_B_trans();

        //We need the REVERSE permutation of C to get back the correct output
        m_perms[0] = mip.get_C_perm();
        runtime_permutation C_perm(m_perms[0].get_order());
        for(size_t i = 0; i < C_perm.get_order(); ++i)
        {
            C_perm[m_perms[0][i]] = i;
        }
        m_C_perm_kern = block_kernel_permute<T>(C_perm);


        m_perms[1] = mip.get_A_perm();
        m_perms[2] = mip.get_B_perm();
        m_ident_perms[0] = runtime_permutation(m_perms[0].get_order());
        m_ident_perms[1] = runtime_permutation(m_perms[1].get_order());
        m_ident_perms[2] = runtime_permutation(m_perms[2].get_order());

        //If we are going to be permuting blocks, we'll need to pre-alloc an array to hold the permuted version
        for(size_t perm_idx = 0; perm_idx < m_perms.size(); ++perm_idx)
        {
            if(m_perms[perm_idx] != m_ident_perms[perm_idx])
            {
                m_perm_kerns[perm_idx] = block_kernel_permute<T>(m_perms[perm_idx]);
                size_t max_block_size = 1;
                for(size_t subspace_idx = 0; subspace_idx < bispaces[perm_idx].get_order(); ++subspace_idx)
                {
                    const subspace& sub = bispaces[perm_idx][subspace_idx];
                    size_t max_block_size_this_dim = 0;
                    for(size_t block_idx = 0; block_idx < sub.get_n_blocks(); ++block_idx)
                    {
                        if(sub.get_block_size(block_idx) > max_block_size_this_dim)
                        {
                            max_block_size_this_dim = sub.get_block_size(block_idx);
                        }
                    }
                    max_block_size *= max_block_size_this_dim;
                }
                m_perm_ptrs[perm_idx] = new T[max_block_size];
                m_perm_ptr_sizes[perm_idx] = max_block_size;
            }
        }
        m_dgemm_fp = dgemm_fp_arr[(size_t)m_A_trans*2+(size_t)m_B_trans];
    }
}

template<typename T>
void libtensor::block_kernel_contract2<T>::init_matvec(const std::vector<sparse_bispace_impl>& bispaces,
                                                       const idx_list& A_contr_inds,
                                                       const idx_list& B_contr_inds)
{
	//Determine the type of matvec mul required to execute the contraction:
	//	A.x
    //	A^T.x
    //	x.A
    //	x.A^T
    //
    //Is the vector the first or second argument in the contraction?
    if(bispaces[1].get_order() == m_n_contr_inds)
    {
        m_vec_bispace_idx = 1;
        m_mat_bispace_idx = 2;
    }
    else
    {
        m_vec_bispace_idx = 2;
        m_mat_bispace_idx = 1;
    }

	//Is A transposed? If so, the contracted indices appear in order at the beginning.
	//Otherwise, they will appear in order at the end
    size_t mat_first_contracted_index = (m_mat_bispace_idx == 1) ? A_contr_inds[0] : B_contr_inds[0];
    m_A_trans = false;
	if(mat_first_contracted_index == 0)
	{
		//Yes
		m_A_trans = true;
        m_dgemv_fn = &linalg::mul2_i_pi_p_x;
	}
    else
    {
        m_dgemv_fn = &linalg::mul2_i_ip_p_x;
    }


    //TODO: Check strided output etc.
}

template<typename T>
void libtensor::block_kernel_contract2<T>::operator ()(
		const std::vector<T*>& ptrs, const std::vector<dim_list>& dim_lists)
{
#ifdef LIBTENSOR_DEBUG
	if(dim_lists.size() != 3 || ptrs.size() != 3)
		throw bad_parameter(g_ns, k_clazz,"operator()(...)",__FILE__, __LINE__, "must pass exactly 3 dim_lists and 3 ptrs");

	for(size_t block_idx = 0; block_idx < dim_lists.size(); ++block_idx)
		if(m_block_orders[block_idx] != dim_lists[block_idx].size())
			throw bad_parameter(g_ns, k_clazz,"operator()(...)",__FILE__, __LINE__, "one or more block dimension lists has wrong size");

    for(size_t i = 0; i < m_ts_groups.size(); ++i)
        for(size_t j = 0; j < m_ts_groups[i].size(); ++j)
            if(dim_lists[m_ts_groups[i][j].first][m_ts_groups[i][j].second] != dim_lists[m_ts_groups[i][0].first][m_ts_groups[i][0].second])
                throw bad_parameter(g_ns, k_clazz,"operator()(...)",__FILE__, __LINE__, "incompatible contraction dimensions");
#endif

    if(m_dgemm_fp == NULL)
    {
        //Matrix-vector mult
        size_t m=1, n=1,lda;

        const dim_list& mat_dim_list = dim_lists[m_mat_bispace_idx];
        T* A = ptrs[m_mat_bispace_idx];
        T* x = ptrs[m_vec_bispace_idx];
        if(m_A_trans)
        {
            for(size_t i = 0; i < m_n_contr_inds; ++i)
            {
                n *= mat_dim_list[i];
            }
            for(size_t i = m_n_contr_inds; i < mat_dim_list.size(); ++i)
            {
                m *= mat_dim_list[i];
            }
            lda = m;
        }
        else
        {
            for(size_t i = 0; i < mat_dim_list.size() - m_n_contr_inds; ++i)
            {
                m *= mat_dim_list[i];
            }
            for(size_t i = mat_dim_list.size() - m_n_contr_inds; i < mat_dim_list.size(); ++i)
            {
                n *= mat_dim_list[i];
            }
            lda = n;
        }

        (*m_dgemv_fn)(NULL,m,n,A,lda,x,1,ptrs[0],1,1.0);
    }
    else
    {
        std::copy(dim_lists.begin(),dim_lists.end(),m_perm_dim_lists.begin());
        //Permute blocks if necessary
        for(size_t perm_idx = 0; perm_idx < m_perms.size(); ++perm_idx)
        {
            if(m_perms[perm_idx] != m_ident_perms[perm_idx])
            {
                m_perms[perm_idx].apply(m_perm_dim_lists[perm_idx]);
                m_perm_kerns[perm_idx].permute(m_perm_ptrs[perm_idx],ptrs[perm_idx],m_perm_dim_lists[perm_idx],dim_lists[perm_idx]);
            }
            else
            {
                m_perm_ptrs[perm_idx] = ptrs[perm_idx];
            }
        }

        //matrix-matrix mult
        size_t m = 1,n = 1,k = 1,lda,ldb,ldc;
        if(m_A_trans)
        {
            for(size_t A_i_idx = m_n_contr_inds; A_i_idx <  m_perm_dim_lists[1].size(); ++A_i_idx)
            {
                m *= m_perm_dim_lists[1][A_i_idx];
            }
            for(size_t k_idx = 0; k_idx < m_n_contr_inds; ++k_idx)
            {
                k *= m_perm_dim_lists[1][k_idx];
            }
            lda = m;
        }
        else
        {
            for(size_t A_i_idx = 0; A_i_idx <  m_block_orders[1] - m_n_contr_inds; ++A_i_idx)
            {
                m *= m_perm_dim_lists[1][A_i_idx];
            }
            for(size_t k_idx = m_block_orders[1] - m_n_contr_inds; k_idx < m_block_orders[1]; ++k_idx)
            {
                k *= m_perm_dim_lists[1][k_idx];
            }
            lda = k;
        }

        if(m_B_trans)
        {
            for(size_t B_j_idx = 0; B_j_idx < m_block_orders[2] - m_n_contr_inds; ++B_j_idx)
            {
                n *= m_perm_dim_lists[2][B_j_idx];
            }
            ldb = k;
        }
        else
        {
            for(size_t B_j_idx = m_n_contr_inds; B_j_idx < m_block_orders[2]; ++B_j_idx)
            {
                n *= m_perm_dim_lists[2][B_j_idx];
            }
            ldb = n;
        }
        ldc = n;

        (*m_dgemm_fp)(NULL,m,n,k,m_perm_ptrs[1],lda,m_perm_ptrs[2],ldb,m_perm_ptrs[0],ldc,1.0);

        if(m_perms[0] != m_ident_perms[0])
        {
            m_C_perm_kern.permute(ptrs[0],m_perm_ptrs[0],dim_lists[0],m_perm_dim_lists[0]);
        }
    }
}

#endif /* BLOCK_KERNEL_CONTRACT2_H */
