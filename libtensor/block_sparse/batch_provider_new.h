#ifndef BATCH_PROVIDER_NEW_H
#define BATCH_PROVIDER_NEW_H

#include "gen_sparse_btensor.h"
#include "batch_kernel_permute.h"
#include "batch_kernel_contract2.h"
#include "batch_kernel_add2.h"
#include "connectivity.h"
#include "../expr/dag/expr_tree.h"
#include "../expr/dag/node_ident.h"
#include "../expr/iface/node_ident_any_tensor.h"
#include "../expr/dag/node_assign.h"
#include "../expr/dag/node_transform.h"
#include "../expr/dag/node_contract.h"
#include "../expr/dag/node_contract.h"
#include "../expr/dag/node_add.h"
#include "../expr/metaprog.h"

namespace libtensor {

template<typename T>
class batch_provider_i
{
public:
    virtual idx_list get_batchable_subspaces() const { return idx_list(); } 
    virtual void get_batch(T* output_ptr) = 0; 
    virtual ~batch_provider_i() {}; 
};

namespace expr {


//Can't do this with a function because C++03 won't let function templates have default template args
template<typename T,size_t NC=0,size_t NA=0,size_t NB=0>
class kernel_builder
{
private:
    const expr_tree& m_tree;
    const expr_tree::edge_list_t& m_edges;
    batch_kernel<T>** m_kern; //Have to cache in a member var to pass through dispatch_1
    std::vector<T*> m_ptrs;
    std::vector<batch_provider_i<T>*> m_suppliers;
    std::vector<sparse_bispace_any_order> m_bispaces;
public:
    std::vector<T*> get_ptrs() { return m_ptrs; }
    std::vector<batch_provider_i<T>*> get_suppliers() { return m_suppliers; }
    std::vector<sparse_bispace_any_order> get_bispaces() { return m_bispaces; }
    void build_kernel(batch_kernel<T>*& kern)
    {
        m_kern = &kern;
        size_t n_tensors_processed = (NC == 0 ? 0 : (NA == 0 ? 1 : (NB == 0 ? 2 : 3)));
        if(n_tensors_processed == m_edges.size() - 1)
        {
            const node& op_node = m_tree.get_vertex(m_edges.back());
            const node& n_0 = m_tree.get_vertex(m_edges[0]);
            const node_ident_any_tensor<NC,T>& n_0_concrete = dynamic_cast< const node_ident_any_tensor<NC,T>& >(n_0);
            gen_sparse_btensor<NC,T>& C = dynamic_cast< gen_sparse_btensor<NC,T>& >(n_0_concrete.get_tensor());
            m_bispaces.push_back(C.get_bispace());
            if(n_tensors_processed > 1)
            {
                const node& n_1 = m_tree.get_vertex(m_edges[1]);
                const node_ident_any_tensor<NA,T>& n_1_concrete = dynamic_cast< const node_ident_any_tensor<NA,T>& >(n_1);
                gen_sparse_btensor<NA,T>& A = dynamic_cast< gen_sparse_btensor<NA,T>& >(n_1_concrete.get_tensor());
                m_ptrs[1] = (T*)A.get_data_ptr();
                m_suppliers[1] = A.get_batch_provider();
                m_bispaces.push_back(A.get_bispace());

                if(n_tensors_processed > 2)
                {
                    const node& n_2 = m_tree.get_vertex(m_edges[2]);
                    const node_ident_any_tensor<NB,T>& n_2_concrete = dynamic_cast< const node_ident_any_tensor<NB,T>& >(n_2);
                    gen_sparse_btensor<NB,T>& B = dynamic_cast< gen_sparse_btensor<NB,T>& >(n_2_concrete.get_tensor());
                    m_ptrs[2] = (T*)B.get_data_ptr();
                    m_suppliers[2] = B.get_batch_provider();
                    m_bispaces.push_back(B.get_bispace());

                    if(op_node.check_type<node_contract>())
                    {
                        const node_contract& n_c = dynamic_cast< const node_contract& >(op_node);
                        kern = new batch_kernel_contract2<T>(C,A,B,n_c.get_map());
                    }
                    else if(op_node.check_type<node_add>())
                    {
                        const node_add& n_a = dynamic_cast< const node_add& >(op_node);
                        expr_tree::edge_list_t add_children = m_tree.get_edges_out(m_edges.back());
                        T lhs_scalar = 1;
                        T rhs_scalar = 1;
                        for(size_t i = 0; i < 2; ++i)
                        {
                            const node& cur_node = m_tree.get_vertex(add_children[i]);
                            if(cur_node.check_type<node_transform<T> >())
                            {
                                //TODO: NEED TO CONSIDER PERMUTATION HERE!
                                const node_transform<T>& n_tf = dynamic_cast< const node_transform<T>& >(cur_node);
                                T cur_scalar = n_tf.get_coeff().get_coeff(); 
                                if(i == 0) lhs_scalar = cur_scalar;
                                if(i == 1) rhs_scalar = cur_scalar;
                            }
                        }
                        kern = new batch_kernel_add2<T>(C,A,B,lhs_scalar,rhs_scalar);
                    }
                    else
                    {
                        throw bad_parameter(g_ns, "kernel_builder","somemethod",
                                __FILE__, __LINE__, "Invalid node type");
                    }
                }
                else if(op_node.check_type<node_transform_base>())
                {
                    {
                        const node_transform_base& n_tf = dynamic_cast< const node_transform_base& >(op_node);
                        kern = new batch_kernel_permute<T>(C,A,n_tf.get_perm());
                    }
                }
                else
                {
                    throw bad_parameter(g_ns, "kernel_builder","somemethod",
                            __FILE__, __LINE__, "Invalid node type");
                }
            }
        }
        else
        {
            size_t cur_node_idx = (NA != 0 ? 2 : (NC != 0 ? 1 : 0));
            const node& cur_tensor_node = m_tree.get_vertex(m_edges[cur_node_idx]);
            dispatch_1<1,4>::dispatch(*this,cur_tensor_node.get_n());
        }
    }

    kernel_builder(const expr_tree& tree,const expr_tree::edge_list_t& edges) : m_tree(tree),m_edges(edges),m_kern(NULL) 
    {
        m_ptrs.resize(m_edges.size() - 1,NULL);
        m_suppliers.resize(m_edges.size() - 1,NULL);
    }

    template<size_t M>
    void dispatch()
    {
        if(NC == 0)
        {
            kernel_builder<T,M,0,0> kb(m_tree,m_edges);
            kb.build_kernel(*m_kern);
            m_ptrs = kb.get_ptrs();
            m_suppliers = kb.get_suppliers();
            m_bispaces = kb.get_bispaces();
        }
        else if(NA == 0)
        {
            kernel_builder<T,NC,M,0> kb(m_tree,m_edges);
            kb.build_kernel(*m_kern);
            m_ptrs = kb.get_ptrs();
            m_suppliers = kb.get_suppliers();
            m_bispaces = kb.get_bispaces();
        }
        else if(NB == 0)
        {
            kernel_builder<T,NC,NA,M> kb(m_tree,m_edges);
            kb.build_kernel(*m_kern);
            m_ptrs = kb.get_ptrs();
            m_suppliers = kb.get_suppliers();
            m_bispaces = kb.get_bispaces();
        }
    }
};

} // namespace expr


template<typename T>
class batch_provider_new : public batch_provider_i<T>
{
private:
    batch_kernel<T>* m_kern;
    std::vector<T*> m_ptrs;
    std::vector<sparse_bispace_any_order> m_bispaces;
    std::vector<batch_provider_i<T>*> m_suppliers;
    idx_list m_suppliers_allocd;
    idx_list m_batchable_subspaces;
    expr::connectivity m_conn;
public:
    static const char* k_clazz; //!< Class name
    batch_provider_new(const expr::expr_tree& tree); 
    virtual void get_batch(T* output_ptr); 
    virtual idx_list get_batchable_subspaces() const;
    ~batch_provider_new();
};

template<typename T>
const char* batch_provider_new<T>::k_clazz = "batch_provider<T>";

template<typename T>
batch_provider_new<T>::batch_provider_new(const expr::expr_tree& tree) : m_conn(tree)
{
    using namespace expr;
    expr_tree::node_id_t root_id = tree.get_root();
    const node& root = tree.get_vertex(root_id);

    if(!root.check_type<node_assign>())
    {
        throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
            "Invalid root node");
    }

    const expr_tree::edge_list_t& children = tree.get_edges_out(root_id);
    if(children.size() != 2)
    {
        throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
            "Root node does not have right number of children");
    }

    expr_tree::edge_list_t edges = expr_tree::edge_list_t(1,children[0]);
    expr_tree::edge_list_t input_edges = tree.get_edges_out(children[1]);

    //Figure out which inputs correspond to tensors and which are nested expressions
    for(size_t i = 0; i < input_edges.size(); ++i)
    {
        const node& cur_node = tree.get_vertex(input_edges[i]);
        //Right now we add spurious assign nodes for direct tensors
        if(cur_node.check_type<node_ident>())
        {
           edges.push_back(input_edges[i]);
        }
        else
        {
            //If one of our inputs is scalar transformed, we ignore it for now
            //Will deal with it when building kernel
            if(cur_node.check_type<node_transform_base>())
            {
                expr_tree::node_id_t next_inter_node_id = tree.get_edges_out(input_edges[i])[0];
                const node& next_inter_node = tree.get_vertex(next_inter_node_id);
                if(next_inter_node.check_type<node_ident>())
                {
                    edges.push_back(next_inter_node_id);
                }
                else if(next_inter_node.check_type<node_assign>())
                {
                    edges.push_back(tree.get_edges_out(next_inter_node_id)[0]);
                }
                else
                {
                    throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
                            "Unsupported node type");
                }
            }
            else if(cur_node.check_type<node_assign>())
            {
                //Save the tensor output from this assignment
                edges.push_back(tree.get_edges_out(input_edges[i])[0]);
            }
            else
            {
                throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
                        "Unsupported node type");
            }
        }
    }
    //Operation node goes at the end
    edges.push_back(children[1]);
    kernel_builder<T> kb(tree,edges);
    kb.build_kernel(m_kern);
    m_ptrs = kb.get_ptrs();
    m_bispaces = kb.get_bispaces();
    m_suppliers = kb.get_suppliers();

    //Create child batch providers for direct tensors used as inputs to this one 
    for(size_t i = 0; i < input_edges.size(); ++i)
    {
        size_t tensor_idx = i+1;
        if(m_ptrs[tensor_idx] == NULL)
        {
            if(m_suppliers[tensor_idx] == NULL)
            {
                m_suppliers[tensor_idx] = new batch_provider_new<T>(tree.get_subtree(input_edges[i]));
                m_suppliers_allocd.push_back(tensor_idx);
            }
            idx_list child_bs(m_suppliers[tensor_idx]->get_batchable_subspaces());
            for(size_t child_bs_idx = 0; child_bs_idx < child_bs.size(); ++child_bs_idx)
            {
                size_t bs = child_bs[child_bs_idx];
                for(size_t conn_subspace_idx = 0; conn_subspace_idx < m_conn.get_n_conn_subspaces(tensor_idx,bs); ++conn_subspace_idx)
                {
                    idx_pair conn_tensor_subspace = m_conn(tensor_idx,bs,conn_subspace_idx);
                    if(conn_tensor_subspace.first == 0)
                    {
                        m_batchable_subspaces.push_back(conn_tensor_subspace.second);
                    }
                }
            }
            m_ptrs[tensor_idx] = new T[m_bispaces[tensor_idx].get_nnz()];
        }
    }
    sort(m_batchable_subspaces.begin(),m_batchable_subspaces.end());
    unique(m_batchable_subspaces.begin(),m_batchable_subspaces.end());
}

template<typename T>
batch_provider_new<T>::~batch_provider_new()
{
    for(size_t i = 0; i < m_suppliers.size(); ++i)
    {
        if(m_suppliers[i] != NULL)
        {
            delete m_ptrs[i];
            if(find(m_suppliers_allocd.begin(),m_suppliers_allocd.end(),i) != m_suppliers_allocd.end())
            {
                delete m_suppliers[i];
            }
        }
    }
}


template<typename T>
void batch_provider_new<T>::get_batch(T* output_ptr)
{ 
    m_ptrs[0] = output_ptr; 
    for(size_t i = 1; i < m_ptrs.size(); ++i)
    {
        if(m_suppliers[i] != NULL)
        {
            m_suppliers[i]->get_batch(m_ptrs[i]);
        }
    }
    //TODO: REMOVE HACK TO MAKE CONTRACTION WORK!!!
    memset(output_ptr,0,m_bispaces[0].get_nnz()*sizeof(T));
    m_kern->generate_batch(m_ptrs,bispace_batch_map()); 
}

template<typename T>
idx_list batch_provider_new<T>::get_batchable_subspaces() const
{ 
    return m_batchable_subspaces;
}



} // namespace libtensor

#endif /* BATCH_PROVIDER_NEW_H */
