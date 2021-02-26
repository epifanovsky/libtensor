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
private:
    runtime_permutation m_C_perm;
    runtime_permutation m_A_perm;
    runtime_permutation m_B_perm;
    bool m_A_trans;
    bool m_B_trans;

                
public:
    matmul_isomorphism_params(const sparse_loop_list& sll);
    runtime_permutation get_C_perm() { return m_C_perm; }
    runtime_permutation get_A_perm() { return m_A_perm; }
    runtime_permutation get_B_perm() { return m_B_perm; }
    bool get_A_trans() { return m_A_trans; }
    bool get_B_trans() { return m_B_trans; }
};

template<typename T>
const char* matmul_isomorphism_params<T>::k_clazz = "matmul_isomorphism_params<T>";


//Determine the type of matmul required to execute the contraction:
//	A.B     [ C_ij = A_ik B_kj ]
//  A.B^T   [ C_ij = A_ik B_jk ]
//	A^T.B   [ C_ij = A_ki B_kj ]
//  A^T.B^T [ C_ij = A_ki B_jk ]
//
//  If the contraction cannot be executed in this fashion, return the permutations 
//  necessary to make it happen
template<typename T>
matmul_isomorphism_params<T>::matmul_isomorphism_params(const sparse_loop_list& sll) : m_C_perm(0),m_A_perm(0),m_B_perm(0) 
{
    static const char *method = "matmul_isomorphism_params, "
        "args go here";

    std::vector<sparse_bispace_any_order> bispaces = sll.get_bispaces();
    if(bispaces.size() != 3)
    {
        throw bad_parameter(g_ns, k_clazz,"matmul_isomorphism_params(...)",__FILE__, __LINE__, "Only 3-bispace contractions can be handled via matmul at this time");
    }
    
    //Build the connectivity vector
    std::vector<block_loop> loops = sll.get_loops();
    idx_list conn;
    for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
    {
        const sparse_bispace_any_order& bispace = bispaces[bispace_idx];
        for(size_t subspace_idx = 0; subspace_idx < bispace.get_order(); ++subspace_idx)
        {
            bool found_connected_subspace = false;
            for(size_t loop_idx = 0; loop_idx < loops.size(); ++loop_idx)
            {
                const block_loop& cur_loop = loops[loop_idx];
                if(!cur_loop.is_bispace_ignored(bispace_idx) && cur_loop.get_subspace_looped(bispace_idx) == subspace_idx)
                {
                    size_t bispace_off = 0;
                    for(size_t other_bispace_idx = 0; other_bispace_idx < bispaces.size(); ++other_bispace_idx)
                    {
                        const sparse_bispace_any_order& other_bispace = bispaces[other_bispace_idx];
                        if(!cur_loop.is_bispace_ignored(other_bispace_idx))
                        {
                            if(other_bispace_idx != bispace_idx)
                            {
                                found_connected_subspace = true;
                                conn.push_back(bispace_off + cur_loop.get_subspace_looped(other_bispace_idx));
                                break;
                            }
                        }
                        bispace_off += other_bispace.get_order();
                    }
                    break;
                }
            }
            if(!found_connected_subspace)
            {
                throw bad_parameter(g_ns, k_clazz,"matmul_isomorphism_params(...)",__FILE__, __LINE__, "Incomplete sparse_loop_list passed!");
            }
        }
    }
	size_t NC = bispaces[0].get_order();
	size_t NA = bispaces[1].get_order();
	size_t NB = bispaces[2].get_order();
    m_C_perm = runtime_permutation(NC);
    m_A_perm = runtime_permutation(NA);
    m_B_perm = runtime_permutation(NB);
    size_t K = (NA+NB-NC)/2;
    size_t N = NA - K;
    size_t M = NB - K;

    //Copy-pasted from contraction2_align, should merge
    size_t ioa = 0, iob = N, ii = NC;

    idx_list idxa1(NA,0), idxa2(NA,0);
    idx_list idxb1(NB,0), idxb2(NB,0);
    idx_list idxc1(NC,0), idxc2(NC,0);

    //  Build initial index ordering
    for(size_t i = 0; i < NC; i++) {
        size_t j = conn[i] - NC;
        if(j < NA) {
            idxc1[i] = ioa;
            idxa1[j] = ioa;
            ioa++;
        } else {
            j -= NA;
            idxc1[i] = iob;
            idxb1[j] = iob;
            iob++;
        }
    }
    for(size_t i = 0; i < NA; i++) {
        if(conn[NC + i] < NC) continue;
        size_t j = conn[NC + i] - NC - NA;
        idxa1[i] = ii;
        idxb1[j] = ii;
        ii++;
    }

    //  Build matricized index ordering

    size_t iai, iao, ibi, ibo, ica, icb;
    if(idxa1[NA - 1] >= NC) {
        //  Last index in A is an inner index
        iai = NA; iao = N;
    } else {
        //  Last index in A is an outer index
        iai = K; iao = NA;
    }
    if(idxb1[NB - 1] >= NC) {
        //  Last index in B is an inner index
        ibi = NB; ibo = M;
    } else {
        //  Last index in B is an outer index
        ibi = K; ibo = NB;
    }
    if(idxc1[NC - 1] < N) {
        //  Last index in C comes from A
        ica = NC; icb = M;
    } else {
        //  Last index in C comes from B
        ica = N; icb = NC;
    }

    for(size_t i = 0; i < NA; i++) {
        if(idxa1[NA - i - 1] >= NC) {
            idxa2[iai - 1] = idxa1[NA - i - 1];
            iai--;
        } else {
            idxa2[iao - 1] = idxa1[NA - i - 1];
            iao--;
        }
    }
    for(size_t i = 0; i < NB; i++) {
        if(idxb1[NB - i - 1] >= NC) {
            idxb2[ibi - 1] = idxb1[NB - i - 1];
            ibi--;
        } else {
            idxb2[ibo - 1] = idxb1[NB - i - 1];
            ibo--;
        }
    }
    for(size_t i = 0; i < NC; i++) {
        if(idxc1[NC - i - 1] < N) {
            idxc2[ica - 1] = idxc1[NC - i - 1];
            ica--;
        } else {
            idxc2[icb - 1] = idxc1[NC - i - 1];
            icb--;
        }
    }

    bool lasta_i = (idxa2[NA - 1] >= NC);
    bool lastb_i = (idxb2[NB - 1] >= NC);
    bool lastc_a = (idxc2[NC - 1] < N);

    if(lastc_a) {
        if(lasta_i) {
            if(lastb_i) {
                //  C(ji) = A(ik) B(jk)
                for(size_t i = 0; i != N; i++) idxa2[i] = idxc2[M + i];
                for(size_t i = 0; i != M; i++) idxc2[i] = idxb2[i];
                for(size_t i = 0; i != K; i++) idxa2[N + i] = idxb2[M + i];
            } else {
                //  C(ji) = A(ik) B(kj)
                for(size_t i = 0; i != N; i++) idxa2[i] = idxc2[M + i];
                for(size_t i = 0; i != M; i++) idxc2[i] = idxb2[K + i];
                for(size_t i = 0; i != K; i++) idxb2[i] = idxa2[N + i];
            }
        } else {
            if(lastb_i) {
                //  C(ji) = A(ki) B(jk)
                for(size_t i = 0; i != N; i++) idxa2[K + i] = idxc2[M + i];
                for(size_t i = 0; i != M; i++) idxc2[i] = idxb2[i];
                for(size_t i = 0; i != K; i++) idxa2[i] = idxb2[M + i];
            } else {
                //  C(ji) = A(ki) B(kj)
                for(size_t i = 0; i != N; i++) idxa2[K + i] = idxc2[M + i];
                for(size_t i = 0; i != M; i++) idxc2[i] = idxb2[K + i];
                for(size_t i = 0; i != K; i++) idxb2[i] = idxa2[i];
            }
        }
    } else {
        if(lasta_i) {
            if(lastb_i) {
                //  C(ij) = A(ik) B(jk)
                for(size_t i = 0; i != N; i++) idxa2[i] = idxc2[i];
                for(size_t i = 0; i != M; i++) idxb2[i] = idxc2[N + i];
                for(size_t i = 0; i != K; i++) idxa2[N + i] = idxb2[M + i];
            } else {
                //  C(ij) = A(ik) B(kj)
                for(size_t i = 0; i != N; i++) idxc2[i] = idxa2[i];
                for(size_t i = 0; i != M; i++) idxb2[K + i] = idxc2[N + i];
                for(size_t i = 0; i != K; i++) idxb2[i] = idxa2[N + i];
            }
        } else {
            if(lastb_i) {
                //  C(ij) = A(ki) B(jk)
                for(size_t i = 0; i != N; i++) idxc2[i] = idxa2[K + i];
                for(size_t i = 0; i != M; i++) idxb2[i] = idxc2[N + i];
                for(size_t i = 0; i != K; i++) idxa2[i] = idxb2[M + i];
            } else {
                //  C(ij) = A(ki) B(kj)
                for(size_t i = 0; i != N; i++) idxc2[i] = idxa2[K + i];
                for(size_t i = 0; i != M; i++) idxc2[N + i] = idxb2[K + i];
                for(size_t i = 0; i != K; i++) idxb2[i] = idxa2[i];
            }
        }
    }

    //Copied from permutation builder...need to merge
    idx_list* idxs_1[3] = {&idxc1,&idxa1,&idxb1};
    idx_list* idxs_2[3] = {&idxc2,&idxa2,&idxb2};
    runtime_permutation* perm_list[3] = {&m_C_perm,&m_A_perm,&m_B_perm};
    for(size_t perm_idx = 0; perm_idx < 3; ++perm_idx)
    {
        const idx_list& idx1_orig = *idxs_1[perm_idx];
        const idx_list& idx2_orig = *idxs_2[perm_idx];
        runtime_permutation& perm = *perm_list[perm_idx];
        size_t len = idx1_orig.size();

        idx_list idx1(len);
        idx_list idx2(len);
        idx_list map(len,0);
        for(size_t i = 0; i < len; i++) {
            idx1[i] = idx1_orig[i];
            idx2[i] = idx2_orig[i];
            map[i] = i;
        }

        size_t i, j;
        idx_list idx(len);
        for(i = 0; i < len; i++) {

            for(j = i + 1; j < len; j++) {
                if(idx1[i] == idx1[j]) {
                    //  Duplicate object
                    throw bad_parameter(g_ns, k_clazz, method,__FILE__, __LINE__, "idx1");
                }
            }
            for(j = 0; j < len; j++) {
                if(idx2[j] == idx1[i]) {
                    idx[i] = j;
                    break;
                }
            }
            if(j == len) {
                //  Object sets differ
                throw bad_parameter(g_ns, k_clazz, method,__FILE__, __LINE__, "idx2");
            }
        }

        i = 0;
        while(i < len) {
            if(i > idx[i]) {
                perm.permute(map[i], map[idx[i]]);
                j = idx[i];
                idx[i] = idx[j];
                idx[j] = j;
                i = 0;
            } else {
                i++;
            }
        }
        //  The loop above generates the inverse of the permutation
        idx_list idx_cp(N);
        for(size_t i = 0; i < N; i++) idx_cp[i] = perm[i];
        for(size_t i = 0; i < N; i++) perm[idx_cp[i]] = i;
    }

    //Look at what our tensors look like after they are permuted to see what matrix multiply function we should use
    idx_list C_conn(conn.begin(),conn.begin()+NC);
    idx_list A_conn(conn.begin()+NC,conn.begin()+NC+NA);
    idx_list B_conn(conn.begin()+NC+NA,conn.begin()+NC+NA+NB);
    m_C_perm.apply(C_conn);
    m_A_perm.apply(A_conn);
    m_B_perm.apply(B_conn);

    if(C_conn[0] >= NC+NA)
    {
        throw bad_parameter(g_ns, k_clazz,method,__FILE__, __LINE__, 
                "Strided output is unsupported at this time");
    }
    m_A_trans = false;
    m_B_trans = false;
    if(A_conn[0] >= NC)
    {
        m_A_trans = true;
    }
    if(B_conn[0] < NC)
    {
        m_B_trans = true;
    }

    //m_dgemm_fp = dgemm_fp_arr[(size_t)A_trans*2+(size_t)B_trans];
}

} // namespace libtensor

#endif /* BLAS_ISOMORPHISM_H */
