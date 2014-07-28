#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

#include <libtensor/expr/dag/expr_tree.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_assign.h>
#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/expr/dag/node_transform.h>
#include "sparse_defs.h"

namespace libtensor
{

namespace expr
{

class connectivity
{
private:
    std::vector< std::vector< idx_pair_list > > m_conn;
    static const char* k_clazz; //!< Class name

public:
    connectivity(const expr_tree& tree);
    size_t get_n_tensors() const { return m_conn.size(); }
    size_t get_order(size_t tensor_idx) const { return m_conn[tensor_idx].size(); }
    size_t get_n_conn_subspaces(size_t tensor_idx,size_t subspace_idx) const { return m_conn[tensor_idx][subspace_idx].size(); }
    idx_pair operator()(size_t tensor_idx,size_t subspace_idx,size_t conn_subspace_idx) const { return m_conn[tensor_idx][subspace_idx][conn_subspace_idx]; }
};

} // namespace expr

} // namespace libtensor



#endif /* CONNECTIVITY_H */
