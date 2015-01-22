#ifndef MATMUL_ISOMORPHISM_PARAMS_H
#define MATMUL_ISOMORPHISM_PARAMS_H

#include "sparse_bispace.h"
#include "../linalg/linalg.h"

namespace libtensor {

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
    matmul_isomorphism_params(const std::vector<sparse_bispace_impl>& bispaces,
                              const std::vector<idx_pair_list>& ts_groups);
    
    runtime_permutation get_C_perm() { return m_C_perm; }
    runtime_permutation get_A_perm() { return m_A_perm; }
    runtime_permutation get_B_perm() { return m_B_perm; }
    bool get_A_trans() { return m_A_trans; }
    bool get_B_trans() { return m_B_trans; }
};

} // namespace libtensor

#endif /* MATMUL_ISOMORPHISM_PARAMS_H */
