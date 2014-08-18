#ifndef BATCH_PROVIDER_H
#define BATCH_PROVIDER_H

#include "gen_sparse_btensor.h"
#include "batch_kernel_permute.h"
#include "batch_kernel_contract2.h"
#include "batch_kernel_add2.h"
#include "batch_kernel_unblock.h"
#include "batch_kernel_reblock.h"
#include "connectivity.h"
#include "../expr/dag/expr_tree.h"
#include "../expr/dag/node_ident.h"
#include "../expr/iface/node_ident_any_tensor.h"
#include "../expr/dag/node_assign.h"
#include "../expr/dag/node_transform.h"
#include "../expr/dag/node_contract.h"
#include "../expr/dag/node_contract.h"
#include "../expr/dag/node_add.h"
#include "../expr/dag/node_unblock.h"
#include "../expr/dag/node_reblock.h"
#include "../expr/metaprog.h"
#include <algorithm>

namespace libtensor {

template<typename T>
class batch_provider_i
{
public:
    virtual idx_list get_batchable_subspaces() const { return idx_list(); } 
    virtual void get_batch(T* output_ptr,const bispace_batch_map& bbm = bispace_batch_map()) = 0; 
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
        size_t n_tensors_to_process = (m_edges.size() == 2) ? 2 : m_edges.size() - 1;
        if(n_tensors_processed == n_tensors_to_process)
        {
            const node& op_node = m_tree.get_vertex(m_edges.back());
            const node& n_0 = m_tree.get_vertex(m_edges[0]);
            const node_ident_any_tensor<NC,T>& n_0_concrete = dynamic_cast< const node_ident_any_tensor<NC,T>& >(n_0);
            gen_sparse_btensor<NC,T>& C = dynamic_cast< gen_sparse_btensor<NC,T>& >(n_0_concrete.get_tensor());
            m_bispaces.push_back(C.get_bispace());
            if(n_tensors_processed > 1)
            {
                const node& n_1 = (m_edges.size() == 2) ? op_node : m_tree.get_vertex(m_edges[1]);
                const node_ident_any_tensor<NA,T>& n_1_concrete = dynamic_cast< const node_ident_any_tensor<NA,T>& >(n_1);
                gen_sparse_btensor<NA,T>& A = dynamic_cast< gen_sparse_btensor<NA,T>& >(n_1_concrete.get_tensor());
                m_ptrs[1] = (T*)A.get_data_ptr();
                m_suppliers[1] = A.get_batch_provider();
                m_bispaces.push_back(A.get_bispace());
                if(&n_1 == &op_node)
                {
                    //TODO: This is very clunky and requires an extra copy step - should form tensor directly from provider into output pointer
                    idx_list perm_entries;
                    for(size_t i = 0; i < NA; ++i) perm_entries.push_back(i);
                    kern = new batch_kernel_permute<T>(C,A,perm_entries);
                }
                else if(n_tensors_processed > 2)
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
                else
                {
                    if(op_node.check_type<node_transform_base>())
                    {
                        const node_transform_base& n_tf = dynamic_cast< const node_transform_base& >(op_node);
                        kern = new batch_kernel_permute<T>(C,A,n_tf.get_perm());
                    }
                    else if(op_node.check_type<node_unblock>())
                    {
                        const node_unblock& n_un = dynamic_cast< const node_unblock& >(op_node);
                        kern = new batch_kernel_unblock<T>(A.get_bispace(),n_un.get_subspace(),A.get_data_ptr() == NULL);
                    }
                    else if(op_node.check_type<node_reblock>())
                    {
                        const node_reblock& n_re = dynamic_cast< const node_reblock& >(op_node);
                        kern = new batch_kernel_reblock<T>(C.get_bispace(),n_re.get_subspace(),C.get_data_ptr() == NULL);
                    }
                    else
                    {
                        throw bad_parameter(g_ns, "kernel_builder","somemethod",
                                __FILE__, __LINE__, "Invalid node type");
                    }
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
        m_ptrs.resize(std::max<size_t>(2,m_edges.size() - 1),NULL);
        m_suppliers.resize(std::max<size_t>(2,m_edges.size() - 1),NULL);
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
class batch_provider : public batch_provider_i<T>
{
private:
    batch_kernel<T>* m_kern;
    std::vector<T*> m_ptrs;
    std::vector<size_t> m_batch_array_sizes;
    std::vector<sparse_bispace_any_order> m_bispaces;
    std::vector<batch_provider_i<T>*> m_suppliers;
    idx_list m_suppliers_allocd;
    idx_list m_batchable_subspaces;
    idx_list m_batched_subspaces;
    idx_pair_list m_batch_list;
    expr::connectivity m_conn;
    void set_batch_info_internal(std::vector<idx_list>::const_iterator& batched_subspace_grps_it,const idx_pair_list& batch_list);
    void set_batch_array_sizes_internal(std::vector< std::vector<size_t> >::const_iterator& it,bool output_batched);
public:
    static const char* k_clazz; //!< Class name
    batch_provider(const expr::expr_tree& tree); 
    virtual void get_batch(T* output_ptr,const bispace_batch_map& bbm = bispace_batch_map()); 
    virtual idx_list get_batchable_subspaces() const;
    ~batch_provider();
    void set_batch_info(const std::vector<idx_list>& batched_subspace_grps,const idx_pair_list& batch_list);
    void set_batch_array_sizes(const std::vector< std::vector<size_t> >& batch_array_size_grps);
    void get_batched_subspace_grps(std::vector<idx_list>& bs_grps,size_t output_subspace_batched=0) const;
    void get_direct_bispace_grps(std::vector< std::vector<sparse_bispace_any_order> >& direct_bispace_grps) const;
};

template<typename T>
const char* batch_provider<T>::k_clazz = "batch_provider<T>";

template<typename T>
batch_provider<T>::batch_provider(const expr::expr_tree& tree) : m_conn(tree),m_batch_list(1,idx_pair(0,0))
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
    expr_tree::edge_list_t input_assignment_nodes;

    //Figure out which inputs correspond to tensors and which are nested expressions
    for(size_t i = 0; i < input_edges.size(); ++i)
    {
        const node& cur_node = tree.get_vertex(input_edges[i]);
        //Right now we add spurious assign nodes for direct tensors
        if(cur_node.check_type<node_ident>())
        {
           edges.push_back(input_edges[i]);
           //Filler so that we can index directly into this list
           input_assignment_nodes.push_back(-1);
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
                    input_assignment_nodes.push_back(-1);
                }
                else if(next_inter_node.check_type<node_assign>())
                {
                    edges.push_back(tree.get_edges_out(next_inter_node_id)[0]);
                    input_assignment_nodes.push_back(next_inter_node_id);
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
                input_assignment_nodes.push_back(input_edges[i]);
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
    for(size_t i = 0; i < std::max<size_t>(1,input_edges.size()); ++i)
    {
        size_t tensor_idx = i+1;
        if(m_ptrs[tensor_idx] == NULL)
        {
            if(m_suppliers[tensor_idx] == NULL)
            {
                m_suppliers[tensor_idx] = new batch_provider<T>(tree.get_subtree(input_assignment_nodes[i]));
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
        }
    }
    sort(m_batchable_subspaces.begin(),m_batchable_subspaces.end());
    unique(m_batchable_subspaces.begin(),m_batchable_subspaces.end());
}

template<typename T>
batch_provider<T>::~batch_provider()
{
    for(size_t i = 0; i < m_suppliers.size(); ++i)
    {
        if(m_suppliers[i] != NULL)
        {
            if(m_ptrs[i] != NULL) delete m_ptrs[i];
            if(find(m_suppliers_allocd.begin(),m_suppliers_allocd.end(),i) != m_suppliers_allocd.end())
            {
                delete m_suppliers[i];
            }
        }
    }
}


template<typename T>
void batch_provider<T>::get_batch(T* output_ptr,const bispace_batch_map& bbm)
{ 
    m_ptrs[0] = output_ptr; 
    m_kern->init(m_ptrs,bbm);
    bool output_batched  = false;
    for(bispace_batch_map::const_iterator it = bbm.begin(); it != bbm.end(); ++it)
    {
        if(it->first.first == 0)
        {
            output_batched = true;
            m_batch_list = idx_pair_list(1,it->second);
            break;
        }
    }

    //Always have 1 dummy batch_list entry to make code run once
    for(size_t batch_idx = 0; batch_idx < m_batch_list.size(); ++batch_idx)
    {
        idx_pair batch_from_supplier = m_batch_list[batch_idx];
        bispace_batch_map augmented_bbm(bbm);
        idx_list::iterator batched_subspace_it = m_batched_subspaces.begin(); 
        if(output_batched) ++batched_subspace_it; //Advance past the output subspace
        for(size_t supplier_idx = 1; supplier_idx < m_ptrs.size(); ++supplier_idx)
        {
            if(m_suppliers[supplier_idx] != NULL)
            {
                bispace_batch_map supplier_bbm;
                if(m_batch_list[0] != idx_pair(0,0))
                {
                    size_t supplier_subspace_idx = *batched_subspace_it;
                    ++batched_subspace_it;
                    supplier_bbm[idx_pair(0,supplier_subspace_idx)] = batch_from_supplier;
                    augmented_bbm[idx_pair(supplier_idx,supplier_subspace_idx)] = batch_from_supplier;
                }
                if(m_ptrs[supplier_idx] == NULL)
                {
                    size_t batch_array_size;
                    if(m_batch_array_sizes.size() > 0)
                    {
                        batch_array_size = m_batch_array_sizes[supplier_idx];
                    }
                    else
                    {
                        batch_array_size = m_bispaces[supplier_idx].get_nnz();
                    }
                    m_ptrs[supplier_idx] = new T[batch_array_size];
                }
                m_suppliers[supplier_idx]->get_batch(m_ptrs[supplier_idx],supplier_bbm);
            }
        }
        m_kern->generate_batch(m_ptrs,augmented_bbm); 

        //TODO HAXX print unblocked tensor AND blocked tensor
#if 0
        if(m_bispaces[0].get_nnz() == 361 && m_bispaces.size() == 2)
        {
            std::cout << "xxx\n"; 
            //std::cout << m_bispaces[0].get_batch_size(1,bbm.begin()->second) << "\n";
            // std::cout << m_bispaces[1].get_batch_size(1,bbm.begin()->second) << "\n";
            std::cout << "xxx\n"; 
            std::cout << "output\n";
            for(size_t i = 0; i < m_bispaces[0].get_nnz(); ++i)
            {
                std::cout << m_ptrs[0][i] << "\n";
            }
            std::cout << "input\n";
            for(size_t i = 0; i < m_bispaces[1].get_nnz(); ++i)
            {
                std::cout << m_ptrs[1][i] << "\n";
            }
        }
#endif
    }
}

template<typename T>
idx_list batch_provider<T>::get_batchable_subspaces() const
{ 
    return m_batchable_subspaces;
}

template<typename T>
void batch_provider<T>::set_batch_info(const std::vector<idx_list>& batched_subspace_grps,const idx_pair_list& batch_list)
{
    std::vector<idx_list>::const_iterator batched_subspace_grps_it = batched_subspace_grps.begin();
    set_batch_info_internal(batched_subspace_grps_it,batch_list);
}

template<typename T>
void batch_provider<T>::set_batch_array_sizes(const std::vector< std::vector<size_t> >& batch_array_size_grps)
{
    std::vector< std::vector<size_t> >::const_iterator it = batch_array_size_grps.begin();
    set_batch_array_sizes_internal(it,false);
}

template<typename T>
void batch_provider<T>::set_batch_array_sizes_internal(std::vector< std::vector<size_t> >::const_iterator& it,bool output_batched)
{
    if(it->size() > 0)
    {
        m_batch_array_sizes.resize(m_suppliers.size(),0);
        size_t m = 0;
        for(size_t supplier_idx = 0; supplier_idx < m_suppliers.size(); ++supplier_idx)
        {
            if(m_suppliers[supplier_idx] != NULL || (supplier_idx == 0 && output_batched))
            {
                m_batch_array_sizes[supplier_idx] = (*it)[m];
                ++m;
            }
        }
    }
    for(size_t i = 0; i < m_suppliers_allocd.size(); ++i)
    {
        static_cast<batch_provider<T>*>(m_suppliers[m_suppliers_allocd[i]])->set_batch_array_sizes_internal(++it,output_batched ? output_batched : !output_batched);
    }
}

template<typename T>
void batch_provider<T>::set_batch_info_internal(std::vector<idx_list>::const_iterator& batched_subspace_grps_it,const idx_pair_list& batch_list)
{
    if(batched_subspace_grps_it->size() > 0)
    {
        m_batched_subspaces = *batched_subspace_grps_it;
        m_batch_list = batch_list;
    }

    for(size_t i = 0; i < m_suppliers_allocd.size(); ++i)
    {
        static_cast<batch_provider<T>*>(m_suppliers[m_suppliers_allocd[i]])->set_batch_info_internal(++batched_subspace_grps_it,batch_list);
    }
}

template<typename T>
void batch_provider<T>::get_direct_bispace_grps(std::vector< std::vector<sparse_bispace_any_order> >& direct_bispace_grps) const
{
    std::vector<sparse_bispace_any_order> grp;
    if(direct_bispace_grps.size() != 0)
    {
        grp.push_back(m_bispaces[0]);
    }
    for(size_t supplier_idx = 0; supplier_idx < m_suppliers.size(); ++supplier_idx)
    {
        if(m_suppliers[supplier_idx] != NULL)
        {
            grp.push_back(m_bispaces[supplier_idx]);
        }
    }
    direct_bispace_grps.push_back(grp);
    for(size_t supplier_allocd_idx = 0; supplier_allocd_idx < m_suppliers_allocd.size(); ++supplier_allocd_idx)
    {
        size_t supplier_idx = m_suppliers_allocd[supplier_allocd_idx];
        static_cast<batch_provider<T>*>(m_suppliers[supplier_idx])->get_direct_bispace_grps(direct_bispace_grps);
    }
}

template<typename T>
void batch_provider<T>::get_batched_subspace_grps(std::vector<idx_list>& bs_grps,size_t output_subspace_batched) const
{
    size_t fixed_supplier_idx;
    size_t fixed_subspace_idx;
    size_t max_direct_tensors_touched = 0;
    if(bs_grps.size() == 0)
    {
        for(size_t supplier_idx = 0; supplier_idx < m_suppliers.size(); ++supplier_idx)
        {
            if(m_suppliers[supplier_idx] != NULL)
            {
                if(m_suppliers[supplier_idx]->get_batchable_subspaces().size() > 0)
                {
                    fixed_supplier_idx = supplier_idx;
                    fixed_subspace_idx = m_suppliers[supplier_idx]->get_batchable_subspaces()[0];
                    max_direct_tensors_touched = 1;
                    break;
                }
                for(size_t subspace_idx = 0; subspace_idx < m_bispaces[supplier_idx].get_order(); ++subspace_idx)
                {
                    //TODO Haxx - just take the index touching the most direct tensors 
                    size_t n_direct_tensors_touched = 1;
                    for(size_t conn_subspace_idx = 0; conn_subspace_idx < m_conn.get_n_conn_subspaces(supplier_idx,subspace_idx); ++conn_subspace_idx)
                    {
                        idx_pair conn_bas = m_conn(supplier_idx,subspace_idx,conn_subspace_idx);
                        if(m_suppliers[conn_bas.first] != NULL)
                        {
                            ++n_direct_tensors_touched;
                        }
                    }
                    if(n_direct_tensors_touched > max_direct_tensors_touched)
                    {
                        max_direct_tensors_touched = n_direct_tensors_touched;
                        fixed_supplier_idx = supplier_idx;
                        fixed_subspace_idx = subspace_idx;
                    }
                }
            }
        }
    }
    else
    {
        fixed_supplier_idx = 0;
        fixed_subspace_idx = output_subspace_batched;
    }

    idx_list cur_grp;
    idx_list bispace_list;
    if(bs_grps.size() > 0 || max_direct_tensors_touched > 0)
    {
        bispace_list.push_back(fixed_supplier_idx);
        cur_grp.push_back(fixed_subspace_idx);
        for(size_t conn_subspace_idx = 0; conn_subspace_idx < m_conn.get_n_conn_subspaces(fixed_supplier_idx,fixed_subspace_idx); ++conn_subspace_idx)
        {
            idx_pair conn_bas = m_conn(fixed_supplier_idx,fixed_subspace_idx,conn_subspace_idx);
            if(m_suppliers[conn_bas.first] != NULL)
            {
                bispace_list.push_back(conn_bas.first);
                cur_grp.push_back(conn_bas.second);
            }
        }
    }

    size_t n_subspaces_should_be_batched = (bs_grps.size() == 0) ? 0 : 1;
    //TODO: Avoid multibatching with this check
    for(size_t supplier_idx = 0; supplier_idx < m_suppliers.size(); ++supplier_idx)
        if(m_suppliers[supplier_idx] != NULL)
            ++n_subspaces_should_be_batched;

    if(n_subspaces_should_be_batched != cur_grp.size())
    {
        throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
                "Operation requires batching over multiple indices");
    }

    //We must ensure that children are visited in-order
    //batchable_subspaces might make our fixed_supplier_idx 2 even when 1 is also batched
    idx_pair_list bispace_pos_list; 
    idx_list sorted_cur_grp(cur_grp.size());
    for(size_t i = 0; i < bispace_list.size(); ++i)
    {
        bispace_pos_list.push_back(idx_pair(bispace_list[i],i));
    }
    sort(bispace_pos_list.begin(),bispace_pos_list.end());
    for(size_t i = 0; i < bispace_pos_list.size(); ++i)
    {
        sorted_cur_grp[i] = cur_grp[bispace_pos_list[i].second];
    }
    bs_grps.push_back(sorted_cur_grp);


    for(size_t i = 0; i < bispace_pos_list.size(); ++i)
    {
        size_t bispace_idx  = bispace_pos_list[i].first;
        if(find(m_suppliers_allocd.begin(),m_suppliers_allocd.end(),bispace_idx) != m_suppliers_allocd.end())
        {
            static_cast<batch_provider<T>*>(m_suppliers[bispace_idx])->get_batched_subspace_grps(bs_grps,sorted_cur_grp[i]);
        }
    }
}

} // namespace libtensor

#endif /* BATCH_PROVIDER_H */
